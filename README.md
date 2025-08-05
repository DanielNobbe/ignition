## Ignition
Ignition is a lightweight PyTorch Ignite-based model training and evaluation framework.

Some baseline model architectures, datasets, loss functions and training utilities are implemented, and these can be easily expanded by adding them under the relevant directories and to the `setup_x` functions in the `__init__.py` files.

Our aim is to allow configuring experiments fully through config files, and we use OmegaConf and Hydra to achieve this. 

Development is ongoing. Generally, we want to increase the number of classes, and allow more configurations, without breaking anything, although we may make breaking changes.

Running the main script can best be done through torchrun:
```
torchrun \
  --nproc_per_node 1 \
  main.py --config-dir=configs \
  --config-name=config.yaml

```