<div align="center">

# ASCENT <!-- no toc -->

Welcome to the code repository for *cardiAc ultrasound Segmentation & Color-dopplEr dealiasiNg Toolbox* (ASCENT).

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code-quality](https://github.com/creatis-myriad/ASCENT/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/creatis-myriad/ASCENT/actions/workflows/code-quality-main.yaml)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/creatis-myriad/ASCENT#LICENSE)

# Publications <!-- no toc -->

[![Journal](https://img.shields.io/badge/IEEE%20TUFFC-2023-4b44ce.svg)](https://doi.org/10.1109/TUFFC.2023.3289621)
[![Paper](http://img.shields.io/badge/paper-arxiv.2306.13695-B31B1B.svg)](https://arxiv.org/abs/2306.13695)

[![Conference](http://img.shields.io/badge/FIMH-2023-4b44ce.svg)](https://doi.org/10.1007/978-3-031-35302-4_25)
[![Paper](http://img.shields.io/badge/paper-arxiv.2305.01997-B31B1B.svg)](https://arxiv.org/abs/2305.01997)

</div>

# Description

ASCENT is a toolbox to segment cardiac structures (left ventricle, right ventricle, etc.) on ultrasound images and perform dealiasing on color Doppler echocardiography. It combines one of the best segmentation framework, [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) with [Lightning](https://lightning.ai/), [Hydra](https://hydra.cc/), and [monai](https://monai.io/). The main reasons of doing so are to take advantage of each library:

- nnUNet's heuristic rules for hyperparameters determination and training scheme give excellent segmentation results.
- Lightning reduces boilerplate and provides better PyTorch code organization.
- Hydra offers pluggable architectures, dynamic configurations, and easy configuration overriding through command lines.
- Monai simplifies the data loading and pre-processing.

For now, ASCENT provides only nnUNet 2D and 3D_fullres architectures (similar to monai's [DynUNet](https://docs.monai.io/en/stable/_modules/monai/networks/nets/dynunet.html)). You can easily plug your own models in ASCENT pipeline.

> **Note**
> nnUNet implemented in ASCENT is the V1.

# Table of Contents <!-- no toc -->

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
- [References](#references)

# How to run

ASCENT has been tested on Linux (Ubuntu 20, Red Hat 7.6), macOS and Windows 10/11.

> **Note**
> Automatic Mixed Precision (AMP) is buggy on Windows devices, e.g. Nan in loss computation. For Windows users, it is recommended to disable it during the run by adding trainer.precision=32 to the train/evaluate/predict command to avoid errors.

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone https://github.com/creatis-myriad/ASCENT
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
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install -e .
   ```

Several new commands will be added to the virtual environment once the installation is completed. These commands all start with `ascent_`.

## Data

Before doing any preprocessing and training, you must first reformat the dataset to the appropriate format and place the converted dataset in the `data/` folder.
Here is an example of the converted [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) dataset using this [conversion script](ascent/dataset_conversion/camus.py).

> **Note**
> Refer to [here](#define-custom-data-and-logs-path) if you want to have a different `data/` folder location.

The reformatted dataset should look like this:

```
data/
├── CAMUS_challenge
│   ├──raw/
│   │  ├──imagesTr/
│   │  │  ├──patient0001_2CH_ED_0000.nii.gz
│   │  │  ├──patient0001_2CH_ES_0000.nii.gz
│   │  │  ├──patient0001_4CH_ED_0000.nii.gz
│   │  │  └── ...
│   │  │
│   │  ├──labelsTr/
│   │  │  ├──patient0001_2CH_ED.nii.gz
│   │  │  ├──patient0001_2CH_ES.nii.gz
│   │  │  ├──patient0001_4CH_ED.nii.gz
│   │  │  └── ...
│   │  │
│   │  ├──imagesTs/
│   │  ├──labelsTs/
│   │  ├──dataset.json
```

More details can be found in [nnUNet's dataset conversion instructions](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1/documentation/dataset_conversion.md).

## Important note

ASCENT uses Hydra to handle the configurations and runs. To know more about Hydra's CLI, refer to its [documentation](https://hydra.cc/docs/intro/).

## Preprocess

ASCENT preprocesses and determines the optimized hyperparameters according to nnUNet's heuristic rules. After placing the dataset in the correct `data/` folder, you can run the preprocessing and planning with the following command:

```bash
ascent_preprocess_and_plan dataset=XXX
```

XXX refers to the dataset name, e.g. CAMUS_challenge.

It is possible to preprocess multiple dataset at once using the `--multirun` flag of Hydra:

```bash
ascent_preprocess_and_plan --multirun dataset=XXX,YYY,ZZZ
```

By default, ASCENT creates a 2D and a 3D experiment configuration files that includes the batch size, U-Net architecture, patch size, etc. You can choose to plan only 2D or 3D experiment by overriding `pl2d` or `pl3d` like follows:

```bash
ascent_preprocess_and_plan dataset=XXX pl2d=False
```

Once `ascent_preprocess_and_plan` is completed, the cropped and preprocessed data will be located respectively at `data/XXX/cropped` and `data/XXX/preprocessed`. New config files are also generated in [ascent/configs/experiment/](ascent/configs/experiment/), [ascent/configs/datamodule/](ascent/configs/datamodule/), and [ascent/configs/model/](ascent/configs/model/). These configs files are named as `XXX_2d.yaml` or `XXX_3d.yaml`, depending on the requested planner(s).

You may override other configurations as long as they are listed in [ascent/configs/preprocess_and_plan.yaml](ascent/configs/preprocess_and_plan.yaml). You can also run the following command to display all available configurations:

```bash
ascent_preprocess_and_plan -h
```

## Model training

With the preprocessing being done, you can now train the model. For all experiments, ASCENT automatically detects the presence of GPU and utilize the GPU if it is available. ASCENTS creates 5-Fold cross validations with train/validation/test splits with 0.8/0.1/0.1/ ratio. You can disable the test splits by overriding `datamodule.test_splits=False`.

Below is an example to train a 2D model on CAMUS dataset with the pre-determined hyperparameters:

```bash
ascent_train experiment=camus_challenge_2d logger=tensorboard

# train on cpu
ascent_train experiment=camus_challenge_2d trainer.accelerator=cpu logger=tensorboard
```

You can override any parameter from command line like this:

```bash
ascent_train experiment=camus_challenge_2d trainer.max_epochs=20 datamodule.batch_size=8 logger=tensorboard
```

If you want to check if all the configurations are correct without running the experiment, simply run:

```bash
ascent_train experiment=camus_challenge_2d --cfg --resolve
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
ascent_train experiment=camus_challenge_2d test=False logger=tensorboard
```

> **Note**
> Refer to [here](#define-custom-data-and-logs-path) if you want to have a different `logs/` folder location.

## Model evaluation

If you skipped the evaluation on validation or test data during, you may evaluate your model afterwards by specifying the `fold` and the `ckpt_path` using `ascent_evaluate`:

```bash
ascent_evaluate experiment=camus_challenge_2d  fold=0 ckpt_path="/path/to/ckpt" logger=tensorboard
```

This will create a new output directory containing the prediction folder.

## Run inference

To run inference on unseen data, you may use the `ascent_predict`:

```bash
ascent_predict dataset=CAMUS_challenge model=camus_challenge_2d ckpt_path=/path/to/checkpoint input_folder=/path/to/input/folder/ output_folder=/path/to/output/folder
```

By default, ASCENT applies test time augmentation during inference. To disable this, override `tta=False`. If you wish to save the predicted softmax probabilities as well, activate the `save_npz=True` flag.

## Experiment tracking

ASCENTS supports all the logging frameworks proposed by PyTorch Lightning: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

For nnUNet experiments, Weights&Biases logger is used by default. This requires you to create an account. After signing up, rename the [.env.example](.env.example) file to `.env` and specify your WANDB API Key as follows:

```bash
### API keys ###
WANDB_API_KEY=<your-wandb-api-key>
```

The environment variables in the `.env` file is automatically loaded by `pyrootutils` before each run.

You can simply override `logger` to use your logger of preference:

```bash
# to use the default tensorboard logger of PyTorch Lightning
ascent_train experiment=camus_challenge_2d logger=tensorboard
```

## Define custom data and logs path

In some cases, you may want to specify your own data and logs paths instead of using the default `data/` and `logs/`. You can do this by setting them in environments variables after renaming the [.env.example](.env.example) file to `.env`. In the `.env`, simply override:

```bash
# custom data path
ASCENT_DATA_PATH="path/to/data"

# custom logs path
ASCENT_LOGS_PATH="paths/to/logs"
```

After that, you must override `paths=custom` in all your commands, e.g.:

```bash
# to use custom data and logs paths
ascent_train experiment=camus_challenge_2d paths=custom
```

# Resources

This project was inspired by:

- [MIC-DKFZ/nnUNetV1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1)
- [Project-MONAI/tutorials/modules/dynunet_pipeline](https://github.com/Project-MONAI/tutorials/tree/main/modules/dynunet_pipeline)
- [NVIDIA/DeepLearningExamples/PyTorch/Segmentation/nnUNet/](https://github.com/NVIDIA/DeepLearningExamples/tree/ddbcd54056e8d1bc1c4d5a8ab34cb570ebea1947/PyTorch/Segmentation/nnUNet)
- [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

# References

If you find this repository useful, please consider citing the paper implemented in this repository relevant to you from the
list below:

```bibtex
@article{ling_dealiasing_2023,
   title = {Phase {Unwrapping} of {Color} {Doppler} {Echocardiography} using {Deep} {Learning}},
   doi = {10.1109/TUFFC.2023.3289621},
   journal = {IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control},
   author = {Ling, Hang Jung and Bernard, Olivier and Ducros, Nicolas and Garcia, Damien},
   month = aug,
   year = {2023},
   pages = {810--820},
}

@inproceedings{ling_temporal_2023,
   title = {Extraction of {Volumetric} {Indices} from {Echocardiography}: {Which} {Deep} {Learning} {Solution} for {Clinical} {Use}?},
   doi = {10.1007/978-3-031-35302-4_25},
   series = {Lecture {Notes} in {Computer} {Science}},
   booktitle = {Functional {Imaging} and {Modeling} of the {Heart}},
   publisher = {Springer Nature Switzerland},
   author = {Ling, Hang Jung and Painchaud, Nathan and Courand, Pierre-Yves and Jodoin, Pierre-Marc and Garcia, Damien and Bernard, Olivier},
   month = june,
   year = {2023},
   pages = {245--254},
}
```
