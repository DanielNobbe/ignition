import sys
from functools import partial
from pprint import pformat
from typing import Any, cast
from warnings import warn

import hydra
import ignite.distributed as idist
import monai
from ignite.engine import Events
from ignite.handlers import LRScheduler, ProgressBar
from ignite.utils import manual_seed
from omegaconf import DictConfig

from ignition.data.utils import denormalize
from ignition.datasets import setup_dataset
from ignition.engines import setup_evaluator, setup_trainer
from ignition.handlers import setup_handlers
from ignition.losses import setup_loss
from ignition.lr_schedulers import setup_lr_scheduler
from ignition.metrics import setup_metrics
from ignition.models import setup_model
from ignition.optimizers import setup_optimizer
from ignition.utils import *

try:
    from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as PyTorchLRScheduler

OmegaConf.register_new_resolver("math", simple_math_resolver)


def train(config: DictConfig):
    """Main training function.
    Performs training logic."""

    world_size = idist.get_world_size()
    if world_size > 1:
        warn(
            "Running this training script on multiple GPUs is not fully supported. To add support, look into adding rank-conditional logic for the handlers, metrics and dataloaders. And verify that all LR schedulers and optimizers support distributed training, e.g. by using `ignite.distributed.auto_optim`."
        )

    if config.task != "segmentation":
        raise ValueError(f"Task type {config.task} is not supported. Only 'segmentation' is currently implemented.")

    monai.config.print_config()
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)

    config.output_dir = output_dir

    # setup basic logger
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))

    # load datasets and create dataloaders
    dataset = setup_dataset(config)
    dataloader_train = dataset.get_train_dataloader()
    dataloader_val = dataset.get_val_dataloader()
    le = len(dataloader_train)

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config))

    if config.get("pretrained") is not None:
        load_pretrained_weights(config, model, logger)

    optimizer = idist.auto_optim(setup_optimizer(model.parameters(), config))
    loss_fn = setup_loss(config).to(device=device)
    lr_scheduler = setup_lr_scheduler(optimizer, config, le)

    # trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, loss_fn, device, dataset, setup_metrics(config, "train", loss_fn, model)
    )
    validator = setup_evaluator(config, model, setup_metrics(config, "val", loss_fn, model), device, dataset, name="val")

    # setup engines logger with python logging
    trainer.logger = validator.logger = logger

    setup_handlers(config, model, optimizer, trainer, validator, lr_scheduler=lr_scheduler)

    logger.info("Start training for %d epochs", config.max_epochs)

    # setup if done. let's run the training
    if config.engine_type == "ignite":
        trainer.run(
            dataloader_train,
            max_epochs=config.max_epochs,
        )
    elif config.engine_type in ["monai", "vista3d"]:
        trainer.run()
    else:
        raise ValueError(f"Unknown engine type: {config.engine_type}. Supported types are 'ignite', 'monai' and 'vista3d.")
    
    logger.info("Training completed successfully!")


def evaluate(config: DictConfig):
    """Run model on evaluation dataset.
    
    Should have the option to store all inferences, and give accurate metrics.

    TODO: Look into loading from any random model checkpoint + spec, without config
    """

    world_size = idist.get_world_size()
    if world_size > 1:
        warn(
            "Running this evaluation script on multiple GPUs is not fully supported. To add support, look into adding rank-conditional logic for the handlers, metrics and dataloaders. And verify that all LR schedulers and optimizers support distributed training, e.g. by using `ignite.distributed.auto_optim`."
        )

    model_dir = config.get("model_dir", None)
    if model_dir is None:
        raise ValueError("For evaluation, the 'model_dir' field must be specified in the config, pointing to the training output directory.")
    
    model_config = get_model_config(model_dir)
    
    monai.config.print_config()
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)
    config.output_dir = output_dir

    # setup basic logger
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))

    dataset = setup_dataset(config)  # Very important to not use the model config here, since the model is trained on that!

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(model_config))
    # model = setup_model(model_config)
    # load model weights from best checkpoint
    model = load_checkpoint_for_evaluation(config, model_dir, model, logger)

    # load model-specific loss
    if "loss" in config.keys():
        loss_fn = setup_loss(config).to(device=device)
    elif "loss" in model_config.keys():
        loss_fn = setup_loss(model_config).to(device=device)
    else:
        loss_fn = None
        logger.warning("No loss function found in either the evaluation or model config. Some metrics may not work.")
    
    evaluator = setup_evaluator(config, model, setup_metrics(config, loss_fn=loss_fn, model=model), device, dataset, output_dir=output_dir / "output")  # name is for post transforms

    evaluator.logger = logger

    logger.info("Starting evaluation.")

    # TODO: Add things to log all images (post transform), and any other scores

    if config.engine_type == "ignite":
        evaluator.run(dataset.get_dataloader())
    elif config.engine_type in ["monai", "vista3d"]:
        evaluator.run()
    else:
        raise ValueError(f"Unknown engine type: {config.engine_type}. Supported types are 'ignite' and 'monai'.")
    
    logger.info("Evaluation complete.")

def run(local_rank: int, config: Any):
    if config.mode == "train":
        train(config)
    elif config.mode == "eval":
        evaluate(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}. Supported modes are 'train'.")


# main entrypoint
@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config.yaml",
)
def main(cfg: DictConfig):

    if cfg.get("resume") is not None:
        
        cfg = resume_from_log(cfg.resume)
        print(f"Resuming from {cfg.resume}")

    config = setup_config(cfg)
    with idist.Parallel(config.backend) as p:
        p.run(run, config=config)



if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")
    main()
