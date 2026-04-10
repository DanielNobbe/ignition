import sys
import os
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
from omegaconf import OmegaConf

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

    print("Initialising training...")

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
    logger = setup_logging(config)  # only logs on rank 0

    logger.info("Configuration: \n%s", pformat(config))


    device = idist.device()

    if config.mode == "train":
        model = idist.auto_model(
            setup_model(config)
        )
    elif config.mode == "finetune":
        if config.finetune.get("model_type") != "ignition":
            raise ValueError(f"Finetuning for model type {config.finetune.model_type} is not supported. Only 'ignition' is currently implemented.")
        model_dir = config.finetune.get("model_dir", None)
        if model_dir is None:
            raise ValueError("For finetuning, the 'model_dir' field must be specified in the config, pointing to the training output directory of the model to finetune.")
        model_config = get_model_config(model_dir)
        override_config_with_model_config(config, model_config, logger)
        model = idist.auto_model(setup_model(model_config))
        model = load_checkpoint_for_evaluation(config, model_dir, model, logger, strip_compiled=True, strip_ddp=True)  # we need to strip the compiled and ddp prefixes in case we're loading a checkpoint that was trained with those. Model needs to be fully loaded before we apply PEFT.

        if config.finetune.get("peft", False):
            from src.ignition.models.peft import try_out_peft
            model = try_out_peft(model)
        else:
            logger.warning("Finetuning without PEFT. This will finetune all parameters, which may not be desired.")

    if config.get("compile", False):
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile.")

    if config.get("pretrained") is not None:
        load_pretrained_weights(config, model, logger)

    # load datasets and create dataloaders
    dataset = setup_dataset(config)
    le = len(dataset.get_train_dataset())
    
    if config.get('debug', False) and rank == 0:
        dataloader_train = dataset.get_train_dataloader()
        dataloader_val = dataset.get_val_dataloader()
        first_batch = next(iter(dataloader_train))
        logger.info(f"First training batch keys: {list(first_batch.keys())}")
        first_val_batch = next(iter(dataloader_val))
        logger.info(f"First validation batch keys: {list(first_val_batch.keys())}")

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
            idist.auto_dataloader(dataset.get_train_dataset()),
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

    if model_config.mode == "finetune" and model_config.finetune.get("peft", False):
        raise ValueError("Evaluation of finetuned models with PEFT is not currently supported. Please implement loading the PEFT adapter.")

    logger = setup_logging(config)  # only logs on rank 0

    # override settings (roi_size, spacing, num_classes) from model_config
    override_config_with_model_config(config, model_config, logger)
    
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
    
    evaluator = setup_evaluator(config, model, setup_metrics(config, loss_fn=loss_fn, model=model), device, dataset, output_dir=os.path.join(output_dir, "output"))  # name is for post transforms

    evaluator.logger = logger

    logger.info("Starting evaluation.")

    # TODO: Add things to log all images (post transform), and any other scores

    if config.engine_type == "ignite":
        from monai.data.utils import list_data_collate
        evaluator.run(idist.auto_dataloader(dataset.get_dataset(), collate_fn=list_data_collate))
    elif config.engine_type in ["monai", "vista3d"]:
        evaluator.run()
    else:
        raise ValueError(f"Unknown engine type: {config.engine_type}. Supported types are 'ignite' and 'monai'.")
    
    logger.info("Evaluation complete.")

def run(config: Any):
    """Run training or evaluation based on config mode."""
    if config.mode in ["train", "finetune"]:
        train(config)
    elif config.mode == "eval":
        evaluate(config)
    else:
        raise ValueError(f"Unknown mode: {config.mode}. Supported modes are 'train', 'finetune' and 'eval'.")


# main entrypoint
@hydra.main(
    version_base=None,
    config_path="configs_v2",
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    if cfg.get("resume") is not None:
        
        cfg = resume_from_log(cfg.resume)
        print(f"Resuming from {cfg.resume}")

    config = setup_config(cfg)

    # Initialize distributed only for real torchrun launches.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    has_torchrun_env = all(
        key in os.environ
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    )

    if has_torchrun_env and world_size > 1:
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"Running on {world_size} processes.")
        idist.initialize(config.backend)  # NOTE: We can't use idist.Parallel, it conflicts with Hydra

        # Now run your training function
        run(config)

        idist.finalize()
    else:
        if world_size > 1 and not has_torchrun_env:
            print("Ignoring partial distributed environment and running single process.")
        print("Running on single process.")
        run(config)



if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")

    main()
