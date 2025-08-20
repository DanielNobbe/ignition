## Ignition
Ignition is a lightweight Ignite-based model training framework, with a main focus on supporting training for MONAI models.  



As it stands, it fully supports MONAI 3D segmentation models, and it has some support for vanilla Ignite models built-in, but the latter functionality is not verified.

Our aim is to allow configuring (and tracking) experiments fully through config files, and we use OmegaConf and Hydra to achieve this.  
Many classes can be instantiated from the config, for instance for transformations or models.  
Custom models, handlers, transforms, etc. can be easily implemented by adding them under the relevant directories. Depending on the implementation, it may be possible to instantiate them directly from the config (we use hydra.instantiate, using a `_target_` key), or by adding it explicitly to the `setup_x` functions in the `__init__.py` files. Make sure that the custom model is still specified through the config file.  

Running the main script can best be done through torchrun:
```
torchrun \
  --nproc_per_node 1 \
  main.py --config-dir=configs \
  --config-name=config.yaml

```

The main script can also be ran from python, or through vscode debugging, see .vscode/launch.json.

Note that we do not currently support multi-gpu training, this requires some tweaks to how the MONAI handlers are set up (e.g., logging should only be done from rank 0). 


## Resume interrupted run
To resume from an interrupted run, the train checkpoint needs to be loaded. Make sure you have a train checkpointer set up in your config. 

Then use the following command:
```
torchrun \
  --nproc_per_node 1 \
  main.py +resume=logs/toy_training_20250820-111755-backend-gloo

```

Note that this produces a new log directory for the new run, but the config file will reference the old run.