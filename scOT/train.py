"""
This script trains a scOT or pretrains Poseidon on a PDE dataset.
Can be also used for finetuning Poseidon.
Can be used in a single config or sweep setup.
"""

import argparse
import torch
import wandb
import numpy as np
import random
import json
import psutil
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
import matplotlib.pyplot as plt
import transformers
from accelerate.utils import broadcast_object_list
from scOT.trainer import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from scOT.model import ScOT, ScOTConfig
from mpl_toolkits.axes_grid1 import ImageGrid
from scOT.problems.base import get_dataset, BaseTimeDataset
from scOT.utils import get_num_parameters, read_cli, get_num_parameters_no_embed
from scOT.metrics import relative_lp_error

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


MODEL_MAP = {
    "T": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [4, 4, 4, 4],
        "embed_dim": 48,
    },
    "S": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 48,
    },
    "B": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 96,
    },
    "L": {
        "num_heads": [3, 6, 12, 24],
        "skip_connections": [2, 2, 2, 0],
        "window_size": 16,
        "patch_size": 4,
        "mlp_ratio": 4.0,
        "depths": [8, 8, 8, 8],
        "embed_dim": 192,
    },
}


def create_predictions_plot(predictions, labels, wandb_prefix):
    assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 4)

    predictions = predictions[indices]
    labels = labels[indices]

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1] + labels.shape[1], 4), axes_pad=0.1
    )

    vmax, vmin = max(predictions.max(), labels.max()), min(
        predictions.min(), labels.min()
    )

    for _i, ax in enumerate(grid):
        i = _i // 4
        j = _i % 4

        if i % 2 == 0:
            ax.imshow(
                predictions[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        else:
            ax.imshow(
                labels[j, i // 2, :, :],
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )

        ax.set_xticks([])
        ax.set_yticks([])

    wandb.log({wandb_prefix + "/predictions": wandb.Image(fig)})
    plt.close()


def setup(params, model_map=True):
    config = None
    RANK = int(os.environ.get("LOCAL_RANK", -1))
    CPU_CORES = len(psutil.Process().cpu_affinity())
    CPU_CORES = min(CPU_CORES, 16)
    print(f"Detected {CPU_CORES} CPU cores, will use {CPU_CORES} workers.")
    if params.disable_tqdm:
        transformers.utils.logging.disable_progress_bar()
    if params.json_config:
        config = json.loads(params.config)
    else:
        config = params.config

    if RANK == 0 or RANK == -1:
        run = wandb.init(
            project=params.wandb_project_name, name=params.wandb_run_name, config=config
        )
        config = wandb.config
    else:

        def clean_yaml(config):
            d = {}
            for key, inner_dict in config.items():
                d[key] = inner_dict["value"]
            return d

        if not params.json_config:
            with open(params.config, "r") as s:
                config = yaml.safe_load(s)
            config = clean_yaml(config)
        run = None

    ckpt_dir = "./"
    if RANK == 0 or RANK == -1:
        if run.sweep_id is not None:
            ckpt_dir = (
                params.checkpoint_path
                + "/"
                + run.project
                + "/"
                + run.sweep_id
                + "/"
                + run.name
            )
        else:
            ckpt_dir = params.checkpoint_path + "/" + run.project + "/" + run.name
    if (RANK == 0 or RANK == -1) and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ls = broadcast_object_list([ckpt_dir], from_process=0)
    ckpt_dir = ls[0]

    if model_map and (
        type(config["model_name"]) == str and config["model_name"] in MODEL_MAP.keys()
    ):
        config = {**config, **MODEL_MAP[config["model_name"]]}
        if RANK == 0 or RANK == -1:
            wandb.config.update(MODEL_MAP[config["model_name"]], allow_val_change=True)

    return run, config, ckpt_dir, RANK, CPU_CORES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train scOT or pretrain Poseidon.")
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument(
        "--finetune_from",
        type=str,
        default=None,
        help="Set this to a str pointing to a HF Hub model checkpoint or a directory with a scOT checkpoint if you want to finetune.",
    )
    parser.add_argument(
        "--replace_embedding_recovery",
        action="store_true",
        help="Set this if you have to replace the embeddings and recovery layers because you are not just using the density, velocity and pressure channels. Only relevant for finetuning.",
    )
    params = read_cli(parser).parse_args()
    run, config, ckpt_dir, RANK, CPU_CORES = setup(params)

    train_eval_set_kwargs = (
        {"just_velocities": True}
        if ("incompressible" in config["dataset"]) and params.just_velocities
        else {}
    )
    if params.move_data is not None:
        train_eval_set_kwargs["move_to_local_scratch"] = params.move_data
    if params.max_num_train_time_steps is not None:
        train_eval_set_kwargs["max_num_time_steps"] = params.max_num_train_time_steps
    if params.train_time_step_size is not None:
        train_eval_set_kwargs["time_step_size"] = params.train_time_step_size
    if params.train_small_time_transition:
        train_eval_set_kwargs["allowed_time_transitions"] = [1]
    train_dataset = get_dataset(
        dataset=config["dataset"],
        which="train",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )
    eval_dataset = get_dataset(
        dataset=config["dataset"],
        which="val",
        num_trajectories=config["num_trajectories"],
        data_path=params.data_path,
        **train_eval_set_kwargs,
    )

    config["effective_train_set_size"] = len(train_dataset)
    time_involved = isinstance(train_dataset, BaseTimeDataset) or (
        isinstance(train_dataset, torch.utils.data.ConcatDataset)
        and isinstance(train_dataset.datasets[0], BaseTimeDataset)
    )

    if not isinstance(train_dataset, torch.utils.data.ConcatDataset):
        resolution = train_dataset.resolution
        input_dim = train_dataset.input_dim
        output_dim = train_dataset.output_dim
        channel_slice_list = train_dataset.channel_slice_list
        printable_channel_description = train_dataset.printable_channel_description
    else:
        resolution = train_dataset.datasets[0].resolution
        input_dim = train_dataset.datasets[0].input_dim
        output_dim = train_dataset.datasets[0].output_dim
        channel_slice_list = train_dataset.datasets[0].channel_slice_list
        printable_channel_description = train_dataset.datasets[
            0
        ].printable_channel_description

    model_config = (
        ScOTConfig(
            image_size=resolution,
            patch_size=config["patch_size"],
            num_channels=input_dim,
            num_out_channels=output_dim,
            embed_dim=config["embed_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            skip_connections=config["skip_connections"],
            window_size=config["window_size"],
            mlp_ratio=config["mlp_ratio"],
            qkv_bias=True,
            hidden_dropout_prob=0.0,  # default
            attention_probs_dropout_prob=0.0,  # default
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=channel_slice_list,
            residual_model="convnext",
            use_conditioning=time_involved,
            learn_residual=False,
        )
        if params.finetune_from is None or params.replace_embedding_recovery
        else None
    )

    train_config = TrainingArguments(
        output_dir=ckpt_dir,
        overwrite_output_dir=True,  #! OVERWRITE THIS DIRECTORY IN CASE, also for resuming training
        evaluation_strategy="epoch",
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        eval_accumulation_steps=16,
        max_grad_norm=config["max_grad_norm"],
        num_train_epochs=config["num_epochs"],
        optim="adamw_torch",
        learning_rate=config["lr"],
        learning_rate_embedding_recovery=(
            None
            if (params.finetune_from is None or "lr_embedding_recovery" not in config)
            else config["lr_embedding_recovery"]
        ),
        learning_rate_time_embedding=(
            None
            if (params.finetune_from is None or "lr_time_embedding" not in config)
            else config["lr_time_embedding"]
        ),
        weight_decay=config["weight_decay"],
        adam_beta1=0.9,  # default
        adam_beta2=0.999,  # default
        adam_epsilon=1e-8,  # default
        lr_scheduler_type=config["lr_scheduler"],
        warmup_ratio=config["warmup_ratio"],
        log_level="passive",
        logging_strategy="steps",
        logging_steps=5,
        logging_nan_inf_filter=False,
        save_strategy="epoch",
        save_total_limit=1,
        seed=SEED,
        fp16=False,
        dataloader_num_workers=CPU_CORES,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_pin_memory=True,
        gradient_checkpointing=False,
        auto_find_batch_size=False,
        full_determinism=False,
        torch_compile=False,
        report_to="wandb",
        run_name=params.wandb_run_name,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping_patience"],
        early_stopping_threshold=0.0,  # set no threshold for now
    )

    if params.finetune_from is not None:
        model = ScOT.from_pretrained(
            params.finetune_from, config=model_config, ignore_mismatched_sizes=True
        )
    else:
        model = ScOT(model_config)
    num_params = get_num_parameters(model)
    config["num_params"] = num_params
    num_params_no_embed = get_num_parameters_no_embed(model)
    config["num_params_wout_embed"] = num_params_no_embed
    if RANK == 0 or RANK == -1:
        print(f"Model size: {num_params}")
        print(f"Model size without embeddings: {num_params_no_embed}")

    def compute_metrics(eval_preds):
        channel_list = channel_slice_list

        def get_statistics(errors):
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

        error_statistics = [
            get_statistics(
                relative_lp_error(
                    eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                    eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                    p=1,
                    return_percent=True,
                )
            )
            for i in range(len(channel_list) - 1)
        ]

        if output_dim == 1:
            error_statistics = error_statistics[0]
            return error_statistics
        else:
            mean_over_means = np.mean(
                np.array(
                    [stats["mean_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            mean_over_medians = np.mean(
                np.array(
                    [stats["median_relative_l1_error"] for stats in error_statistics]
                ),
                axis=0,
            )
            error_statistics_ = {
                "mean_relative_l1_error": mean_over_means,
                "mean_over_median_relative_l1_error": mean_over_medians,
            }
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[printable_channel_description[i] + "/" + key] = (
                        value
                    )
            return error_statistics_

    trainer = Trainer(
        model=model,
        args=train_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    trainer.train(resume_from_checkpoint=params.resume_training)
    trainer.save_model(train_config.output_dir)
    
    if (RANK == 0 or RANK == -1) and params.push_to_hf_hub is not None:
        model.push_to_hub(params.push_to_hf_hub)

    do_test = (
        True
        if params.max_num_train_time_steps is None
        and params.train_time_step_size is None
        and not params.train_small_time_transition
        and not ".time" in config["dataset"]
        else False
    )
    if do_test:
        print("Testing...")
        test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        out_test_set_kwargs = (
            {"just_velocities": True}
            if ("incompressible" in config["dataset"]) and params.just_velocities
            else {}
        )
        if params.move_data is not None:
            test_set_kwargs["move_to_local_scratch"] = params.move_data
            out_test_set_kwargs["move_to_local_scratch"] = params.move_data
        if time_involved:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 14,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 20,
                "allowed_time_transitions": [1],
            }
        if "RayleighTaylor" in config["dataset"]:
            test_set_kwargs = {
                **test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 7,
                "allowed_time_transitions": [1],
            }
            out_test_set_kwargs = {
                **out_test_set_kwargs,
                "max_num_time_steps": 1,
                "time_step_size": 10,
                "allowed_time_transitions": [1],
            }

        test_dataset = get_dataset(
            dataset=config["dataset"],
            which="test",
            num_trajectories=config["num_trajectories"],
            data_path=params.data_path,
            **test_set_kwargs,
        )
        try:
            out_dist_test_dataset = get_dataset(
                dataset=config["dataset"] + ".out",
                which="test",
                num_trajectories=config["num_trajectories"],
                data_path=params.data_path,
                **out_test_set_kwargs,
            )
        except:
            out_dist_test_dataset = None
        predictions = trainer.predict(test_dataset, metric_key_prefix="")
        if RANK == 0 or RANK == -1:
            metrics = {}
            for key, value in predictions.metrics.items():
                metrics["test/" + key[1:]] = value
            wandb.log(metrics)
            create_predictions_plot(
                predictions.predictions,
                predictions.label_ids,
                wandb_prefix="test",
            )

        # evaluate on out-of-distribution test set
        if out_dist_test_dataset is not None:
            predictions = trainer.predict(out_dist_test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test_out_dist/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test_out_dist",
                )

        if time_involved and (test_set_kwargs["time_step_size"] // 2 > 0):
            trainer.set_ar_steps(test_set_kwargs["time_step_size"] // 2)
            predictions = trainer.predict(test_dataset, metric_key_prefix="")
            if RANK == 0 or RANK == -1:
                metrics = {}
                for key, value in predictions.metrics.items():
                    metrics["test/ar/" + key[1:]] = value
                wandb.log(metrics)
                create_predictions_plot(
                    predictions.predictions,
                    predictions.label_ids,
                    wandb_prefix="test/ar",
                )

            # evaluate on out-of-distribution test set
            if out_dist_test_dataset is not None:
                trainer.set_ar_steps(out_test_set_kwargs["time_step_size"] // 2)
                predictions = trainer.predict(
                    out_dist_test_dataset, metric_key_prefix=""
                )
                if RANK == 0 or RANK == -1:
                    metrics = {}
                    for key, value in predictions.metrics.items():
                        metrics["test_out_dist/ar/" + key[1:]] = value
                    wandb.log(metrics)
                    create_predictions_plot(
                        predictions.predictions,
                        predictions.label_ids,
                        wandb_prefix="test_out_dist/ar",
                    )
