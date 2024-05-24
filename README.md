# Poseidon: Efficient Foundation Models for PDEs

This is the source code for the paper [*Poseidon: Efficient Foundation Models for PDEs*](TODO). It also acts as a package if you want to use the models in your code.

![Poseidon](assets/fig1.png)

Find pretrained models and pretraining dataset in our collection on the [🤗 Hub – Pretrained Models and Pretraining Datasets](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a). All datasets corresponding to downstream tasks can be downloaded from the respective collection on the [🤗 Hub – Downstream Tasks](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14) as well. To use them, follow the respective sections below.

## Usage

### Installation & Requirements

To get all requirements and install the package, run (inside this folder), after getting this repository:

```bash
pip install -e .
```

We recommend running the above command in a [virtual environment](https://docs.python.org/3/library/venv.html).

After installation, you can import the models and use the training and inference scripts from everywhere on your system.

### Using the models in your own code

To use the (pretrained) models in your own code, you can use the following code snippet (after installing):

```python
from scOT.models import ScOT

model = scOT.from_pretrained("camlab-ethz/Poseidon-<MODEL_SIZE>")
```

This will load the pretrained model from the 🤗 Hub. `<MODEL_SIZE>` has to be replaced by `T`, `B`, or `L`, for the respective pretrained model. You can also load a model from a local path by providing the path to the `from_pretrained` method.

To finetune and replace embeddings and recovery parameters, load the model as follows:

```python
from scOT.models import ScOT

model = ScOT.from_pretrained("camlab-ethz/Poseidon-<MODEL_SIZE>", config= model_config, ignore_mismatched_sizes=True)
```

Here, `model_config` is a `ScOTConfig` with the correct input/output dimensions. We also refer to [the training/finetuning script](scOT/train.py), see below on usage, which might be easier.

### Training & Finetuning

The easiest way to finetune **Poseidon** on your own dataset is by plugging in your own dataset and running the provided training script as follows:

```bash
accelerate launch scOT/train.py \
    --config <WANDB_CONFIG_FILE> \
    --wandb_run_name <WANDB_RUN_NAME> \
    --wandb_project_name <WANDB_PROJECT_NAME> \
    --checkpoint_path <CHECKPOINT_PATH> \
    --data_path <DATA_PATH> \
    --finetune_from <PRETRAINED_MODEL> \
    --replace_embedding_recovery <SET ONLY IF EMBED/RECOVERY NEEDS TO BE REPLACED>
```

For more arguments and options, see the help message of the script:
    
```bash
accelerate launch scOT/train.py --help
```

Since the code is built on top of [🤗 Accelerate](https://huggingface.co/docs/accelerate/en/index), you should run `accelerate config` first.

We also make heavy use of [Weights and Biases](wandb.com) to log and organise all our runs. The code might run without it (by setting `WANDB_MODE=disabled`), but we don't give any guarantees as this probably breaks the folder structure.

Most of the actual training configuration is set in a YAML config file (see for all arguments to set for a single W&B [run](configs/run.yaml) or a W&B [sweep](configs/sweep.yaml) (multiple runs, see the [W&B documentation](https://docs.wandb.ai/guides/sweeps) on how to start a sweep)). The config file is passed to the training script via the `--config` argument.

We do our pretrainings with the same script.

### Inference/Testing

To evaluate a model on a dataset, you can use the inference script, for all possible arguments see the help message:

```bash
python -m scOT.inference --help
```

## Datasets

We provide all datasets used in the paper on the 🤗 Hub. You can download them from the respective collections: 
- [🤗 Hub – Pretraining Datasets](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a)
- [🤗 Hub – Downstream Tasks](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14)

### Naming convention in the code

In the code, we refer to the datasets by a different identifier than on the 🤗 Hub, see the following table for a mapping:

| Code Identifier | 🤗 Hub/Paper Identifier |
| ----------------|------------------------- |
|fluids.incompressible.Sines| NS-Sines|
|fluids.incompressible.Gaussians| NS-Gauss|
|fluids.compressible.Riemann|CE-RP|
|fluids.compressible.RiemannCurved|CE-CRP|
|fluids.compressible.KelvinHelmholtz|CE-KH|
|fluids.compressible.Gaussians|CE-Gauss|
|fluids.incompressible.PiecewiseConstants|NS-PwC|
|fluids.incompressible.VortexSheet|NS-SVS|
|fluids.incompressible.BrownianBridge|NS-BB|
|fluids.incompressible.ShearLayer|NS-SL|
|fluids.incomressible.PiecewiseConstants.tracer|NS-Tracer-PwC|
|fluids.incompressible.forcing.KolmogorovFlow|FNS-KF|
|fluids.compressible.RiemannKelvinHelmholtz|CE-RPUI|
|fluids.compressible.RichtmyerMeshkov|CE-RM|
|fluids.compressible.gravity.RayleighTaylor|GCE-RT|
|wave.Layer|Wave-Layer|
|wave.Gaussians|Wave-Gauss|
|reaction_diffusion.AllenCahn|ACE|
|fluids.compressible.steady.Airfoil(.time)|SE-AF|
|elliptic.poisson.Gaussians(.time)|Poisson-Gauss|
|elliptic.Helmholtz(.time)|Helmholtz|

Adding the suffix `.time` to the dataset identifier will load the dataset as time-dependent dataset, i.e. as a long-time limit – use that suffix for finetuning on time-independent datasets.

### Download & Assembly

Download all the datasets used in our paper from the 🤗 Hub. You may want to use the CLI provided by the [Hub Python Library](https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-download):
    
```bash 
huggingface-cli download camlab-ethz/<DATASET IDENTIFIER FROM PAPER> --repo-type dataset --local-dir <LOCAL DIRECTORY>
```

This will download a specific dataset to the specified `LOCAL DIRECTORY`. After download, you need to assemble the datasets to the format expected by the code; for that, we refer to the README in the respective dataset repository. After assembly, remove the chunked dataset files, as they are not needed for training, and place the assembled dataset at the path you specify as `--data_path` for the training/inference script. You may also specify the 🤗 Hub cache location by specifying the environment variable `HF_HOME` as this is where the download will be performed to.

### Adding your own dataset

We encourage adding your own datasets. For that, you can subclass from [BaseDataset and BaseTimeDataset](scOT/problems/base.py) and add it to the `get_dataset` selector method. You can then use the dataset in the training script by specifying the dataset identifier in the config file.

For subclassing, we refer to the docstrings in the base classes and the existing datasets in the [problems](scOT/problems) folder.

## Pretrained models

Pretrained models are available on the 🤗 Hub, see the [Poseidon collection](https://huggingface.co/collections/camlab-ethz/poseidon-664fa125729c53d8607e209a) for all models. You can download them via the 🤗 Hub API or by using the `from_pretrained` method, see above.

## Citation

If you use our models, code, or datasets, please consider citing our paper:

```bibtex
TODO
```
