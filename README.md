[![Code-Generator](https://badgen.net/badge/Template%20by/Code-Generator/ee4c2c?labelColor=eaa700)](https://github.com/pytorch-ignite/code-generator)

# Segmentation Template

This is the segmentation template by Code-Generator using `deeplabv3_resnet101` and `cifar10` dataset from TorchVision and training is powered by PyTorch and PyTorch-Ignite.

**Note:**
The dataset used in this template is quite substantial, with a size of several GBs and automatically downloading it can be seen as an unexpected behaviour. To prevent unexpected behavior or excessive bandwidth usage, the automatic downloading of the dataset has been disabled by default.

To download the dataset:

```python
python -c "from ignition.data import download_datasets; download_datasets('/path/to/data')"

```

or

```py
from ignition.data import download_datasets
download_datasets('/path/to/data')
```

## Getting Started

Install the dependencies with `pip`:

```sh
pip install -r requirements.txt --progress-bar off -U
```

### Code structure

```
|
|- README.md
|
|- main.py : main script to run
|- data.py : helper module with functions to setup input datasets and create dataloaders
|- models.py : helper module with functions to create a model or multiple models
|- trainers.py : helper module with functions to create trainer and evaluator
|- utils.py : module with various helper functions
|- vis.py : helper module for data visualizations
|- requirements.txt : dependencies to install with pip
|
|- config.yaml : global configuration YAML file
|
|- test_all.py : test file with few basic sanity checks
```

## Training

### Multi GPU Training (`torchrun`) (recommended)

```sh
torchrun \
  --nproc_per_node 1 \
  main.py --config-dir=[dir-path] \
  --config-name=[config-name] ++backend= nccl \
  override_arg=[value]
```

Note: We use Hydra with [OmegaConfig](https://omegaconf.readthedocs.io/en/2.3_branch/) as the default argument parser here. For more information check the [Hydra docs](https://hydra.cc)
