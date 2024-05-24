"""
Use this script for inference/testing a scOT model.
The script can be used in different modes:
- save_samples: Save samples from a model.
- save_samples_sweep: Save samples from a sweep.
- eval: Evaluate a model on the test set.
- eval_sweep: Evaluate a sweep on the test set.
- eval_accumulation_error: Evaluate the accumulation error of a model.
- eval_resolutions: Evaluate a model on different resolutions.

See the --help page for more information.
"""

import argparse
import torch
import numpy as np
import random
import psutil
import os
import pandas as pd
import wandb
from transformers.trainer_utils import EvalPrediction
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.metrics import relative_lp_error, lp_error


SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def get_trainer(
    model_path,
    batch_size,
    dataset,
    full_data=False,
    output_all_steps=False,
    workers=-1,
):
    """
    Get a trainer for the model (actually just using the interface for inference).

    Args:
        model_path: str
            Path to the model.
        batch_size: int
            Batch size for evaluation.
        dataset: BaseTimeDataset
            Test set.
        full_data: bool
            Whether to save the full data distribution.
        output_all_steps: bool
            Whether to output all preliminary steps in autoregressive rollout.
        workers: int
            Number of workers for evaluation. If -1 will use all available cores.
    """
    num_cpu_cores = len(psutil.Process().cpu_affinity())
    if workers == -1:
        workers = num_cpu_cores
    if workers > num_cpu_cores:
        workers = num_cpu_cores
    assert workers > 0

    model = ScOT.from_pretrained(model_path)
    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=16,
        dataloader_num_workers=workers,
    )
    time_involved = isinstance(dataset, BaseTimeDataset)

    def compute_metrics(eval_preds):
        if time_involved and output_all_steps:
            return {}
        channel_list = dataset.channel_slice_list

        def get_relative_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_l1_error": median_error,
                "mean_l1_error": mean_error,
                "std_l1_error": std_error,
                "min_l1_error": min_error,
                "max_l1_error": max_error,
            }

        relative_errors = [
            relative_lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
                return_percent=True,
            )
            for i in range(len(channel_list) - 1)
        ]

        errors = [
            lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
            )
            for i in range(len(channel_list) - 1)
        ]

        relative_error_statistics = [
            get_relative_statistics(relative_errors[i])
            for i in range(len(channel_list) - 1)
        ]

        error_statistics = [
            get_statistics(errors[i]) for i in range(len(channel_list) - 1)
        ]

        if dataset.output_dim == 1:
            relative_error_statistics = relative_error_statistics[0]
            error_statistics = error_statistics[0]
            if full_data:
                relative_error_statistics["relative_full_data"] = relative_errors[
                    0
                ].tolist()
                error_statistics["full_data"] = errors[0].tolist()
            return {**relative_error_statistics, **error_statistics}
        else:
            mean_over_relative_means = np.mean(
                np.array(
                    [
                        stats["mean_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_relative_medians = np.mean(
                np.array(
                    [
                        stats["median_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_means = np.mean(
                np.array([stats["mean_l1_error"] for stats in error_statistics]), axis=0
            )
            mean_over_medians = np.mean(
                np.array([stats["median_l1_error"] for stats in error_statistics]),
                axis=0,
            )

            error_statistics_ = {
                "mean_relative_l1_error": mean_over_relative_means,
                "mean_over_median_relative_l1_error": mean_over_relative_medians,
                "mean_l1_error": mean_over_means,
                "mean_over_median_l1_error": mean_over_medians,
            }
            #!! The above is different from train and finetune (here mean_relative_l1_error is mean over medians instead of mean over means)
            for i, stats in enumerate(relative_error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i]
                            + "/"
                            + "relative_full_data"
                        ] = relative_errors[i].tolist()
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i] + "/" + "full_data"
                        ] = errors[i].tolist()
            return error_statistics_

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
    )
    return trainer


def rollout(trainer, dataset, ar_steps=1, output_all_steps=False):
    """
    Do a rollout of the model.

    Args:
        trainer: Trainer
            Trainer for the model.
        dataset: BaseTimeDataset
            Test set.
        ar_steps: int or list
            Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.
        output_all_steps: bool
            Whether to output all preliminary steps in autoregressive rollout.
    """
    time_involved = isinstance(dataset, BaseTimeDataset)
    if time_involved and ar_steps != 1:
        trainer.set_ar_steps(ar_steps, output_all_steps=output_all_steps)
    else:
        trainer.set_ar_steps(ar_steps=1, output_all_steps=False)

    prediction = trainer.predict(dataset, metric_key_prefix="")

    try:
        return prediction.predictions, prediction.label_ids, prediction.metrics
    except:
        return prediction.predictions


def get_test_set(
    dataset, data_path, initial_time=None, final_time=None, dataset_kwargs={}
):
    """
    Get a test set (input at initial_time, output at final_time).

    Args:
        dataset: str
            Dataset name.
        data_path: str
            Path to data.
        initial_time: int
            Initial time step to start from.
        final_time: int
            Final time step to end at.
        dataset_kwargs: dict
            Additional arguments for dataset as in scOT.problems.base.get_dataset.
    """
    if initial_time is not None and final_time is not None:
        dataset_kwargs = {
            **dataset_kwargs,
            "fix_input_to_time_step": initial_time,
            "time_step_size": final_time - initial_time,
            "max_num_time_steps": 1,
        }
    dataset = get_dataset(
        dataset=dataset,
        which="test",
        num_trajectories=1,
        data_path=data_path,
        move_to_local_scratch=None,
        **dataset_kwargs,
    )
    return dataset


def get_first_n_inputs(dataset, n):
    """
    Helper to get the first n inputs of a dataset.
    """
    inputs = []
    for i in range(n):
        inputs.append(dataset[i]["pixel_values"])
    return torch.stack(inputs)


def get_trajectories(
    dataset, data_path, ar_steps, initial_time, final_time, dataset_kwargs
):
    """
    Get full trajectories in a dataset. Helper for accumulation error evaluation.

    Args:
        dataset: str
            Dataset name.
        data_path: str
            Path to data.
        ar_steps: int or list
            Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.
        initial_time: int
            Initial time step to start from.
        final_time: int
            Final time step to end at.
        dataset_kwargs: dict
            Additional arguments for dataset as in scOT.problems.base.get_dataset.
    """
    trajectories = []
    if isinstance(ar_steps, int):
        delta = (final_time - initial_time) // ar_steps
        for i in range(ar_steps):
            dataset_ = get_test_set(
                dataset,
                data_path,
                initial_time + i * delta,
                initial_time + (i + 1) * delta,
                dataset_kwargs,
            )
            traj_ = []
            for j in range(len(dataset_)):
                traj_.append(dataset_[j]["labels"])
            trajectories.append(torch.stack(traj_))
    else:
        running_time = initial_time
        for i in ar_steps:
            dataset_ = get_test_set(
                dataset, data_path, running_time, running_time + i, dataset_kwargs
            )
            running_time += i
            traj_ = []
            for j in range(len(dataset_)):
                traj_.append(dataset_[j]["labels"])
            trajectories.append(torch.stack(traj_))
    return torch.stack(trajectories, dim=1)


def remove_underscore_dict(d):
    return {key[1:] if key.startswith("_") else key: value for key, value in d.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Do different evaluations for a model, see --mode."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Model path. Not required when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="File to load/write to. May also be a directory to save samples.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which test set to load. Not required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Whether to save full data distributions.",
    )
    parser.add_argument(
        "--initial_time",
        type=int,
        default=None,
        help="Initial time step to start from.",
    )
    parser.add_argument(
        "--final_time",
        type=int,
        default=None,
        help="Final time step to end at.",
    )
    parser.add_argument(
        "--ar_steps",
        type=int,
        nargs="+",
        default=[1],
        help="Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "save_samples",
            "save_samples_sweep",
            "eval",
            "eval_sweep",
            "eval_accumulation_error",
            "eval_resolutions",
        ],
        default="eval",
        help="Mode to run. Can be either save_samples to save n samples, save_samples_sweep, eval (to evaluate a single model), eval_sweep (to evaluate all models in a wandb sweep), eval_accumulation_error (to evaluate a model's accumulation error), eval_resolutions (to evaluate a model on different resolutions).",
    )
    parser.add_argument(
        "--save_n_samples",
        type=int,
        default=1,
        help="Number of samples to save. Only required for mode==save_samples or save_samples_sweep.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        help="List of resolutions to evaluate. Only required for mode==eval_resolutions.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="scOT",
        help="Wandb project name. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=False,
        help="Wandb entity name. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--wandb_sweep_id",
        type=str,
        default=None,
        help="Wandb sweep id. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Base checkpoint directory. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--exclude_dataset",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclude from evaluation. Only relevant when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--exclusively_evaluate_dataset",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclusively evaluate. Only relevant when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--just_velocities",
        action="store_true",
        help="Use just velocities in incompressible flow data.",
    )
    parser.add_argument(
        "--allow_failed",
        action="store_true",
        help="Allow failed runs to be taken into account with eval_sweep.",
    )
    parser.add_argument(
        "--append_time",
        action="store_true",
        help="Append .time to dataset name for evaluation.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=128,
        help="Filter runs for number of training trajectories. Only relevant if mode==eval_sweep or save_samples_sweep.",
    )
    params = parser.parse_args()
    if len(params.ar_steps) == 1:
        params.ar_steps = params.ar_steps[0]
        ar_steps = params.ar_steps
    else:
        ar_steps = params.ar_steps
        params.ar_steps = [
            step / (params.final_time - params.initial_time) for step in params.ar_steps
        ]
    dataset_kwargs = {}
    if params.just_velocities:
        dataset_kwargs["just_velocities"] = True
    if params.mode == "save_samples":
        dataset = get_test_set(
            params.dataset,
            params.data_path,
            params.initial_time,
            params.final_time,
            dataset_kwargs,
        )
        trainer = get_trainer(params.model_path, params.batch_size, dataset)
        inputs = get_first_n_inputs(dataset, params.save_n_samples)
        outputs, labels, _ = rollout(trainer, dataset, ar_steps=params.ar_steps)
        np.save(
            params.file + "/" + params.dataset.replace(".", "-") + "/" + "inputs.npy",
            inputs.cpu().numpy(),
        )
        np.save(
            params.file + "/" + params.dataset.replace(".", "-") + "/" + "labels.npy",
            labels[: params.save_n_samples],
        )
        np.save(
            params.file + "/" + params.dataset.replace(".", "-") + "/" + "outputs.npy",
            outputs[: params.save_n_samples],
        )
    elif params.mode == "save_samples_sweep":
        api = wandb.Api()
        sweep = api.sweep(
            params.wandb_entity
            + "/"
            + params.wandb_project
            + "/"
            + params.wandb_sweep_id
        )
        for run in sweep.runs:
            if run.state == "finished" or (
                params.allow_failed and run.state == "failed"
            ):
                dset_name = run.config["dataset"]
                if run.config["num_trajectories"] != params.num_trajectories:
                    continue
                if dset_name in params.exclude_dataset:
                    continue
                if (
                    len(params.exclusively_evaluate_dataset) > 0
                    and dset_name not in params.exclusively_evaluate_dataset
                ):
                    continue
                num_trajectories = run.config["num_trajectories"]
                ckpt_dir = (
                    params.ckpt_dir
                    + "/"
                    + params.wandb_project
                    + "/"
                    + params.wandb_sweep_id
                    + "/"
                    + run.name
                )
                items = os.listdir(ckpt_dir)
                dirs = [
                    item
                    for item in items
                    if os.path.isdir(os.path.join(ckpt_dir, item))
                ]
                if len(dirs) > 1:
                    print(
                        "WARNING: more than one checkpoint in run directory " + ckpt_dir
                    )
                    print("choosing " + dirs[0])
                model_path = os.path.join(ckpt_dir, dirs[0])
                dataset = get_test_set(
                    dset_name,
                    params.data_path,
                    params.initial_time,
                    params.final_time,
                    dataset_kwargs,
                )
                trainer = get_trainer(model_path, params.batch_size, dataset)
                inputs = get_first_n_inputs(dataset, params.save_n_samples)
                outputs, labels, _ = rollout(trainer, dataset, ar_steps=params.ar_steps)
                if not os.path.exists(params.file + "/" + dset_name.replace(".", "-")):
                    os.makedirs(params.file + "/" + dset_name.replace(".", "-"))
                if not os.path.exists(
                    params.file
                    + "/"
                    + dset_name.replace(".", "-")
                    + "/"
                    + str(num_trajectories)
                ):
                    os.makedirs(
                        params.file
                        + "/"
                        + dset_name.replace(".", "-")
                        + "/"
                        + str(num_trajectories)
                    )
                np.save(
                    params.file
                    + "/"
                    + dset_name.replace(".", "-")
                    + "/"
                    + str(num_trajectories)
                    + "/inputs.npy",
                    inputs.cpu().numpy(),
                )
                np.save(
                    params.file
                    + "/"
                    + dset_name.replace(".", "-")
                    + "/"
                    + str(num_trajectories)
                    + "/labels.npy",
                    labels[: params.save_n_samples],
                )
                np.save(
                    params.file
                    + "/"
                    + dset_name.replace(".", "-")
                    + "/"
                    + str(num_trajectories)
                    + "/"
                    + "outputs.npy",
                    outputs[: params.save_n_samples],
                )
    else:
        if params.mode == "eval":
            dataset = get_test_set(
                params.dataset,
                params.data_path,
                params.initial_time,
                params.final_time,
                dataset_kwargs,
            )
            trainer = get_trainer(
                params.model_path,
                params.batch_size,
                dataset,
                full_data=params.full_data,
            )
            _, _, metrics = rollout(
                trainer,
                dataset,
                ar_steps=params.ar_steps,
                output_all_steps=False,
            )
            data = {
                "dataset": params.dataset,
                "initial_time": params.initial_time,
                "final_time": params.final_time,
                "ar_steps": ar_steps,
                **metrics,
            }
            data = [remove_underscore_dict(data)]
        elif params.mode == "eval_sweep":
            api = wandb.Api()
            sweep = api.sweep(
                params.wandb_entity
                + "/"
                + params.wandb_project
                + "/"
                + params.wandb_sweep_id
            )
            data = []
            for run in sweep.runs:
                if run.state == "finished" or (
                    params.allow_failed and run.state == "failed"
                ):
                    dset_name = (
                        run.config["dataset"]
                        if not params.append_time
                        else run.config["dataset"] + ".time"
                    )
                    if dset_name in params.exclude_dataset:
                        continue
                    if (
                        len(params.exclusively_evaluate_dataset) > 0
                        and dset_name not in params.exclusively_evaluate_dataset
                    ):
                        continue
                    num_trajectories = run.config["num_trajectories"]
                    ckpt_dir = (
                        params.ckpt_dir
                        + "/"
                        + params.wandb_project
                        + "/"
                        + params.wandb_sweep_id
                        + "/"
                        + run.name
                    )
                    items = os.listdir(ckpt_dir)
                    dirs = [
                        item
                        for item in items
                        if os.path.isdir(os.path.join(ckpt_dir, item))
                    ]
                    if len(dirs) > 1:
                        print(
                            "WARNING: more than one checkpoint in run directory "
                            + ckpt_dir
                        )
                        print("choosing " + dirs[0])
                        continue
                    if len(dirs) == 0:
                        continue
                    model_path = os.path.join(ckpt_dir, dirs[0])
                    dataset = get_test_set(
                        dset_name,
                        params.data_path,
                        params.initial_time,
                        params.final_time,
                        dataset_kwargs,
                    )
                    trainer = get_trainer(
                        model_path,
                        params.batch_size,
                        dataset,
                        full_data=params.full_data,
                    )
                    _, _, metrics = rollout(
                        trainer,
                        dataset,
                        ar_steps=params.ar_steps,
                        output_all_steps=False,
                    )
                    data.append(
                        remove_underscore_dict(
                            {
                                "dataset": dset_name,
                                "num_trajectories": num_trajectories,
                                "initial_time": params.initial_time,
                                "final_time": params.final_time,
                                "ar_steps": ar_steps,
                                **metrics,
                            }
                        )
                    )
        elif params.mode == "eval_accumulation_error":
            dataset = get_test_set(
                params.dataset,
                params.data_path,
                params.initial_time,
                params.final_time,
                dataset_kwargs,
            )
            trainer = get_trainer(
                params.model_path,
                params.batch_size,
                dataset,
                output_all_steps=True,
                full_data=params.full_data,
            )
            predictions, _, _ = rollout(
                trainer,
                dataset,
                ar_steps=params.ar_steps,
                output_all_steps=True,
            )
            labels = get_trajectories(
                params.dataset,
                params.data_path,
                params.ar_steps,
                params.initial_time,
                params.final_time,
                dataset_kwargs,
            )

            def compute_metrics(eval_preds):
                channel_list = dataset.channel_slice_list

                def get_relative_statistics(errors):
                    median_error = np.median(errors, axis=0)
                    mean_error = np.mean(errors, axis=0)
                    std_error = np.std(errors, axis=0)
                    min_error = np.min(errors, axis=0)
                    max_error = np.max(errors, axis=0)
                    return {
                        "median_relative_l1_error": median_error,
                        "mean_relative_l1_error": mean_error,
                        "std_relative_l1_error": std_error,
                        "min_relative_l1_error": min_error,
                        "max_relative_l1_error": max_error,
                    }

                def get_statistics(errors):
                    median_error = np.median(errors, axis=0)
                    mean_error = np.mean(errors, axis=0)
                    std_error = np.std(errors, axis=0)
                    min_error = np.min(errors, axis=0)
                    max_error = np.max(errors, axis=0)
                    return {
                        "median_l1_error": median_error,
                        "mean_l1_error": mean_error,
                        "std_l1_error": std_error,
                        "min_l1_error": min_error,
                        "max_l1_error": max_error,
                    }

                relative_errors = [
                    relative_lp_error(
                        eval_preds.predictions[
                            :, channel_list[i] : channel_list[i + 1]
                        ],
                        eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                        p=1,
                        return_percent=True,
                    )
                    for i in range(len(channel_list) - 1)
                ]

                errors = [
                    lp_error(
                        eval_preds.predictions[
                            :, channel_list[i] : channel_list[i + 1]
                        ],
                        eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                        p=1,
                    )
                    for i in range(len(channel_list) - 1)
                ]

                relative_error_statistics = [
                    get_relative_statistics(relative_errors[i])
                    for i in range(len(channel_list) - 1)
                ]

                error_statistics = [
                    get_statistics(errors[i]) for i in range(len(channel_list) - 1)
                ]

                if dataset.output_dim == 1:
                    relative_error_statistics = relative_error_statistics[0]
                    error_statistics = error_statistics[0]
                    if params.full_data:
                        relative_error_statistics["relative_full_data"] = (
                            relative_errors[0].tolist()
                        )
                        error_statistics["full_data"] = errors[0].tolist()
                    return {**relative_error_statistics, **error_statistics}
                else:
                    mean_over_relative_means = np.mean(
                        np.array(
                            [
                                stats["mean_relative_l1_error"]
                                for stats in relative_error_statistics
                            ]
                        ),
                        axis=0,
                    )
                    mean_over_relative_medians = np.mean(
                        np.array(
                            [
                                stats["median_relative_l1_error"]
                                for stats in relative_error_statistics
                            ]
                        ),
                        axis=0,
                    )
                    mean_over_means = np.mean(
                        np.array(
                            [stats["mean_l1_error"] for stats in error_statistics]
                        ),
                        axis=0,
                    )
                    mean_over_medians = np.mean(
                        np.array(
                            [stats["median_l1_error"] for stats in error_statistics]
                        ),
                        axis=0,
                    )

                    error_statistics_ = {
                        "mean_relative_l1_error": mean_over_relative_means,
                        "mean_over_median_relative_l1_error": mean_over_relative_medians,
                        "mean_l1_error": mean_over_means,
                        "mean_over_median_l1_error": mean_over_medians,
                    }
                    #!! The above is different from train and finetune (here mean_relative_l1_error is mean over medians instead of mean over means)
                    for i, stats in enumerate(relative_error_statistics):
                        for key, value in stats.items():
                            error_statistics_[
                                dataset.printable_channel_description[i] + "/" + key
                            ] = value
                            if params.full_data:
                                error_statistics_[
                                    dataset.printable_channel_description[i]
                                    + "/"
                                    + "relative_full_data"
                                ] = relative_errors[i].tolist()
                    for i, stats in enumerate(error_statistics):
                        for key, value in stats.items():
                            error_statistics_[
                                dataset.printable_channel_description[i] + "/" + key
                            ] = value
                            if params.full_data:
                                error_statistics_[
                                    dataset.printable_channel_description[i]
                                    + "/"
                                    + "full_data"
                                ] = errors[i].tolist()
                    return error_statistics_

            data = []
            for step in range(predictions.shape[1]):
                metrics = compute_metrics(
                    EvalPrediction(predictions[:, step], labels[:, step].cpu().numpy())
                )
                if isinstance(params.ar_steps, int):
                    delta = (params.final_time - params.initial_time) // params.ar_steps
                else:
                    delta = params.ar_steps[step]
                data.append(
                    remove_underscore_dict(
                        {
                            "dataset": params.dataset,
                            "initial_time": params.initial_time + step * delta,
                            "final_time": params.initial_time + (step + 1) * delta,
                            **metrics,
                        }
                    )
                )
        elif params.mode == "eval_resolutions":
            data = []
            for resolution in params.resolutions:
                dataset_kwargs = {"resolution": resolution}
                dataset = get_test_set(
                    params.dataset,
                    params.data_path,
                    params.initial_time,
                    params.final_time,
                    dataset_kwargs,
                )
                trainer = get_trainer(
                    params.model_path,
                    params.batch_size,
                    dataset,
                    full_data=params.full_data,
                )
                _, _, metrics = rollout(
                    trainer,
                    dataset,
                    ar_steps=params.ar_steps,
                    output_all_steps=False,
                )
                data.append(
                    remove_underscore_dict(
                        {
                            "dataset": params.dataset,
                            "initial_time": params.initial_time,
                            "final_time": params.final_time,
                            "ar_steps": ar_steps,
                            "resolution": resolution,
                            **metrics,
                        }
                    )
                )

        if os.path.exists(params.file):
            df = pd.read_csv(params.file)
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(params.file, index=False)
