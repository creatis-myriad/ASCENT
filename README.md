<div align="center">

# ASCENT <!-- no toc -->

Welcome to the code repository for *cardiAc ultrasound Segmentation & Color-dopplEr dealiasiNg Toolbox* (ASCENT).

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code-quality](https://github.com/HangJung97/ASCENT/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/HangJung97/ASCENT/actions/workflows/code-quality-main.yaml)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/HangJung97/ASCENT#LICENSE)

</div>

# Description

ASCENT is a toolbox to segment cardiac structures (left ventricle, right ventricle, etc.) on ultrasound images and perform dealiasing on color Doppler echocardiography. It combines one of the best segmentation framework, [nnUNet](https://github.com/MIC-DKFZ/nnUNet) with [PyTorch Lightning](https://www.pytorchlightning.ai/), [Hydra](https://hydra.cc/), and [monai](https://monai.io/). The main reasons of doing so are to take advantage of each library:

- nnUNet's heuristic rules for hyperparameters determination and training scheme give excellent segmentation results.
- PyTorch Lightning reduces boilerplate and provides better PyTorch code organization.
- Hydra offers pluggable architectures, dynamic configurations, and easy configuration overriding through command lines.
- Monai simplifies the data loading and pre-processing.

For now, ASCENT provides only nnUNet 2D and 3D_fullres architectures (similar to monai's [DynUNet](https://docs.monai.io/en/stable/_modules/monai/networks/nets/dynunet.html)). You can easily plug your own models in ASCENT pipeline.

# Table of Contents <!-- no toc -->

- [Description](#description)
- [How to run](#how-to-run)
  - [Install](#install)
  - [Data](#data)
  - [Important note](#important-note)
  - [Preprocess](#preprocess)
  - [Model training](#model-training)
  - [Model evaluation](#model-evaluation)
  - [Run inference](#run-inference)
  - [Experiment tracking](#experiment-tracking)
  - [Define custom data and logs path](#define-custom-data-and-logs-path)
- [Resources](#resources)

# How to run

ASCENT has been tested on Linux (Ubuntu 20, Red Hat 7.6), macOS and Windows 10.

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone https://github.com/HangJung97/ASCENT
   cd ASCENT
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n ascent python=3.10
   conda activate ascent
   ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to instructions. Grab the one with GPU for faster training:
   ```bash
   # example for linux or Windows
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install -e .
   ```

Several new commands will be added to the virtual environment once the installation is completed. These commands all start with `ascent_`.

## Data

Next, download the [Camus](https://www.creatis.insa-lyon.fr/Challenge/camus/) dataset. You have to reformat the dataset to new format and place the converted dataset in the `data/` folder.

> **Note**
> Refer to [here](#define-custom-data-and-logs-path) if you want to have a different `data/` folder location.

The reformatted dataset should look like this:

```
data/
├── CAMUS
│   ├──raw/
│   │  ├──imagesTr/
│   │  │  ├──CAMUS_0001_0000.nii.gz
│   │  │  ├──CAMUS_0002_0000.nii.gz
│   │  │  ├──CAMUS_0003_0000.nii.gz
│   │  │  └── ...
│   │  │
│   │  ├──labelsTr/
│   │  │  ├──CAMUS_0001.nii.gz
│   │  │  ├──CAMUS_0002.nii.gz
│   │  │  ├──CAMUS_0003.nii.gz
│   │  │  └── ...
│   │  │
│   │  ├──imagesTs/
│   │  ├──labelsTs/
│   │  ├──dataset.json
```

More details can be found in [nnUNet's dataset conversion instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md).

## Important note

ASCENT uses Hydra to handle the configurations and runs. To know more about Hydra's CLI, refer to its [documentation](https://hydra.cc/docs/intro/).

## Preprocess

ASCENT preprocesses and determines the optimized hyperparameters according to nnUNet's heuristic rules. After placing the dataset in the correct `data/` folder, you can run the preprocessing and planning with the following command:

```bash
ascent_preprocess_and_plan dataset=XXX
```

XXX refers to the dataset name, e.g. CAMUS.

It is possible to preprocess multiple dataset at once using the `--multirun` flag of Hydra:

```bash
ascent_preprocess_and_plan --multirun dataset=XXX,YYY,ZZZ
```

By default, ASCENT creates a 2D and a 3D experiment configuration files that includes the batch size, U-Net architecture, patch size, etc. You can choose to plan only 2D or 3D experiment by overriding `pl2d` or `pl3d` like follows:

```bash
ascent_preprocess_and_plan dataset=XXX pl2d=False
```

Once `ascent_preprocess_and_plan` is completed, the cropped and preprocessed data will be located respectively at `data/XXX/cropped` and `data/XXX/preprocessed`. New config files are also generated in [configs/experiment/](configs/experiment/), [configs/datamodule/](configs/datamodule/), and [configs/model/](configs/model/). These configs files are named as `XXX_2d.yaml` or `XXX_3d.yaml`, depending on the requested planner(s).

You may override other configurations as long as they are listed in [configs/preprocess_and_plan.yaml](configs/preprocess_and_plan.yaml). You can also run the following command to display all available configurations:

```bash
ascent_preprocess_and_plan -h
```

## Model training

With the preprocessing being done, you can now train the model. For all experiments, ASCENT automatically detects the presence of GPU and utilize the GPU if it is available. ASCENTS creates 5-Fold cross validations with train/validation/test splits with 0.8/0.1/0.1/ ratio. You can disable the test splits by overriding `datamodule.test_splits=False`.

Below is an example to train a 2D model on CAMUS dataset with the pre-determined hyperparameters:

```bash
ascent_train experiment=camus_2d logger=tensorboard

# train on cpu
ascent_train experiment=camus_2d trainer.accelerator=cpu logger=tensorboard
```

You can override any parameter from command line like this:

```bash
ascent_train experiment=camus_2d trainer.max_epochs=20 datamodule.batch_size=8 logger=tensorboard
```

If you want to check if all the configurations are correct without running the experiment, simply run:

```bash
ascent_train experiment=camus_2d --cfg --resolve
```

Hydra creates new output directory for every executed run. Default ASCENT's nnUNet logging structure is as follows:

```
├── logs
│   ├── dataset_name
│   │   ├── nnUNet
│   │   │   ├── 2D                          # nnUNet variant (2D or 3D)
│   │   │   │   ├── Fold_X                      # Fold to train on
│   │   │   │   │   ├── runs                        # Logs generated by single runs
│   │   │   │   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   │   │   │   └── ...
│   │   │   │   │   │
│   │   │   │   │   └── multiruns                   # Logs generated by multiruns
│   │   │   │   │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │   │   │   │       │   ├──1                        # Multirun job number
│   │   │   │   │       │   ├──2
│   │   │   │   │       │   └── ...
│   │   │   │   │       └── ...
```

At the end of the training, the prediction on validation or test data will be executed and saved in the output folder, named as `validation_raw` or `testing_raw`. To disable this, override `test=False`:

```bash
ascent_train experiment=camus_2d test=False logger=tensorboard
```

> **Note**
> Refer to [here](#define-custom-data-and-logs-path) if you want to have a different `logs/` folder location.

## Model evaluation

If you skipped the evaluation on validation or test data during, you may evaluate your model afterwards by specifying the `fold` and the `ckpt_path` using `ascent_evaluate`:

```bash
ascent_evaluate experiment=camus_2d  fold=0 ckpt_path="/path/to/ckpt" logger=tensorboard
```

This will create a new output directory containing the prediction folder.

## Run inference

To run inference on unseen data, you may use the `ascent_predict`:

```bash
ascent_predict dataset=CAMUS model=camus_2d ckpt_path=/path/to/checkpoint input_folder=/path/to/input/folder/ output_folder=/path/to/output/folder
```

By default, ASCENT applies test time augmentation during inference. To disable this, override `tta=False`. If you wish to save the predicted softmax probabilities as well, activate the `save_npz=True` flag.

## Experiment tracking

ASCENTS supports all the logging frameworks proposed by PyTorch Lightning: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

For nnUNet experiments, Comet logger is used by default. This requires you to create an account. After signing up, rename the [.env.example](.env.example) file to `.env` and specify your Comet API Key as follows:

```bash
### API keys ###
COMET_API_KEY=<your-comet-api-key>
```

The environment variables in the `.env` file is automatically loaded by pyrootutils before each run.

Override `logger` to use your logger of preference:

```bash
# to use the default tensorboard logger of PyTorch Lightning
ascent_train experiment=camus_2d logger=tensorboard
```

## Define custom data and logs path

In some cases, you may want to specify your own data and logs paths instead of using the default `data/` and `logs/`. You can do this by setting them in environments variables after renaming the [.env.example](.env.example) file to `.env`. In the `.env`, simply override:

```bash
# custom data path
DATA_PATH="path/to/data"

# custom logs path
LOGS_PATH="paths/to/logs"
```

After that, you must override `paths=custom` in all your commands, e.g.:

```bash
# to use custom data and logs paths
ascent_train experiment=camus_2d paths=custom
```

# Resources

This project was inspired by:

- [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
- [Project-MONAI/tutorials/modules/dynunet_pipeline](https://github.com/Project-MONAI/tutorials/tree/main/modules/dynunet_pipeline)
- [NVIDIA/DeepLearningExamples/PyTorch/Segmentation/nnUNet/](https://github.com/NVIDIA/DeepLearningExamples/tree/ddbcd54056e8d1bc1c4d5a8ab34cb570ebea1947/PyTorch/Segmentation/nnUNet)
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)
