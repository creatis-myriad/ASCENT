<div align="center">

# ASCENT

Welcome to the code repository for *cardiAc ultrasound Segmentation & Color-dopplEr dealiasiNg Toolbox* (ASCENT).

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code-quality](https://github.com/HangJung97/ASCENT/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/HangJung97/ASCENT/actions/workflows/code-quality-main.yaml)

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/HangJung97/ASCENT#LICENSE)

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/HangJung97/ASCENT
cd ASCENT

# [OPTIONAL] create conda environment
conda create -n ascent python=3.10
conda activate ascent

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install the project
pip install .
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# train on CPU
ascent_train experiment=camus_2d trainer.accelerator=cpu

# train on GPU
ascent_train experiment=camus_2d trainer.accelerator=gpu
```

You can override any parameter from command line like this

```bash
ascent_train experiment=camus_2d trainer.max_epochs=20 datamodule.batch_size=8
```
