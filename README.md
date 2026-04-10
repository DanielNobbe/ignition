## Ignition
Ignition is a lightweight Ignite-based training framework for volumetric (medical) segmentation. It is built using the MONAI framework, using its models, trainers and transforms, and offers some support for generic Ignite/PyTorch models.

Our aim is for users to only write config files, which define the model and training parameters, based on a good set of defaults. 

### Installation
For installation, please clone the repo and use 

### Usage
For training a model, see as an example the `configs/train-toy-dataset.yaml` file, which can be ran using the `sh/train-toy.sh` file (which is set up for HPC clusters, most of the environment variables are superflous when running on an isolated machine).

Some settings to tweak:
- `roi_size`: The toy dataset is made up of small crops, for larger images we recommend removing the `model.blocks_down` override, and setting the `roi_size` to the smallest size larger than the median size of your images that is divisible by 32 in very dim. Optionally, the override can be kept with an extra 2 in the middle for roi sizes divisble by 16. Generally, a larger roi size results in better models, although there are diminishing returns.
- `spacing`: For the `spacing` parameter, we suggest to set the median spacing of your dataset.
- `learning_rate`: We suggest to increase the learning rate until collapse occurs, and choose a value slightly below collapse.
- `lr_scheduler`: The default lr sheduler is a cosine annealing one with warmup, but we have seen good results using the `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` scheduler, to nudge the model to a slightly better optimum after a couple of restarts.
- `dataset.split_type`: Our dataset constructor supports not just datalists, but also JSON files that are just a global list (or id->datapoint dicts) with random splitting, or such lists with folds specified for each item (`split_type: cross_val` and `cross_val_fold: int`). We will add examples in a future commit.
- `dataset.transforms.train`: Any relevant transforms from MONAI can be included. For debugging, we include the `ignition.transforms.Debugd` transform, which sets a `pdb` breakpoint, where the user can inspect the `data` dict. It requires the `keys` parameter with a list of strings, the exact strings do not matter. Note that this requires `num_workers` to be set to 0.
- Multiple GPUs: If multiple GPUs are available, the bash script `sh/train-toy.sh` will detect them and spawn multiple processes through torchrun. Please adjust the batch size as needed, and monitor GPU memory usage to tweak. Multi-node training is not supported at the moment but should only require minimal adjustments to the launch script.



### Tracking training
By default, Weights&Biases and Tensorboard logging are enabled. For W&B, you need to login through the CLI first. For tensorboard, use the command `tensorboard --logdir logs/` to start the server. Only Tensorboard currently logs validation inputs and outputs. 

Ignition is a lightweight Ignite-based model training framework, with a main focus on supporting training for MONAI models.  


### Resuming
It is possible to resume training from a terminated training run, where the resumption inherits all settings from the run. It is possible to modify some settings of the terminated run by editing its config lock file. Example:

```
torchrun \
  --nproc_per_node 1 \
  main.py +resume=logs/toy_training_20250820-111755-backend-gloo

```

Note that this produces a new log directory for the new run, but the config file will reference the old run.

### Finetuning
Finetuning similarly can pick up from a completed run. It requires the following addition to the main config file:

```
mode: finetune
finetune:
  model_type: ignition
  base_model: <some-run-name>
  model_dir: logs/${finetune.base_model}
  peft: false
```

### Evaluation
To use a model for inference, use the 'evaluation' mode. See the example `configs/evaluate_toy.yaml`. All predictions, along with the transformed images and labels are saved in the `results` directory.

If labels are included in the dataset, these will be used to compute some metrics. These metrics are correct and reliable, but we suggest to use an external library to metric calculations when comparing to other model frameworks. In Dice score calculations, there are some subtleties that may differ between frameworks, such as how foreground/background count, and how spacing affects the numbers.

If there are no labels included, only inference is done. It is possible to use the `MonaiEvalSegmentationFolder` dataset type with an `images_dir` specified to run inference on a directory of images. 