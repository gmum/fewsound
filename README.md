# FewSound

Repository based on https://github.com/WUT-AI/hypersound

## Setup

Setup conda environment:

```console
conda env create -f environment.yml
```
Set environmental variables in the environment, for example:

```console
conda env config vars set DATA_DIR=~/datasets
conda env config vars set RESULTS_DIR=~/results
conda env config vars set WANDB_ENTITY=my_wandb_entity
conda env config vars set WANDB_PROJECT=fewsound
```

Make sure that `pytorch-yard` is using the appropriate version (defined in `train.py`). If not, then correct package version with something like:

```console
pip install --force-reinstall pytorch-yard==2022.9.1
```

## Experiments

Default experiment:

```console
python train.py
```

Custom settings:

```console
python train.py cfg.learning_rate=0.01 cfg.pl.max_epochs=100
```

Isolated training of a target network on a single recording:

```console
python train_inr.py cfg.pl.max_epochs=100 cfg.inr_audio_path=PATH_TO_WAV
```
