import os
from typing import Any

import torch.nn as nn
from flatten_dict import flatten
from hydra.utils import instantiate
from ignite.contrib.engines import common
from ignite.engine import Engine

from omegaconf import DictConfig, ListConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
from tensorboardX import SummaryWriter

from ignition.utils import get_epoch_function

from ignite.handlers.wandb_logger import WandBLogger
from ignite.engine.events import Events
import ignite.distributed as idist

from ignition.wandb.utils import setup_wandb_logging

"""
train_handlers:


LrScheduleHandler
ValidationHandler
StatsHandler
TensorBoardStatsHandler
CheckpointSaver
ProgressBar

TODO: Create function to create these handlers,
then migrate to hydra instantiate


We can either:
- create a list of handlers and pass them to the trainer at init
or
- attach all handlers to the trainer after init

The second option is better, since the earlystop handler needs to be attached to the evaluator but also needs to have a trainer attached
"""

from logging import getLogger

printer = getLogger(__name__)


def setup_handlers(
    config: dict,
    model: nn.Module,
    optimizer: Optimizer,
    trainer: Engine,
    validator: Engine,
    lr_scheduler: LrScheduler | None = None,
):
    """Setup all handlers for the trainer and validator."""
    if idist.get_rank() == 0:
        writer = setup_tensorboard_writer(config, trainer, optimizer, validator)
    else:
        writer = None  # only log from rank 0
    setup_wandb_logger(config, trainer, validator, optimizer)

    setup_train_handlers(
        config,
        model,
        optimizer,
        writer,
        trainer,
        validator,
        lr_scheduler=lr_scheduler,
    )

    setup_validation_handlers(
        config,
        model,
        trainer,
        writer,
        validator,
    )

    # TODO: Customise tensorboard loggers more, to include e.g. epoch number. Should be fairly easy to unpack what common.setup_tb_logging does


def dict2mdtable(d, key="Name", val="Value"):
    rows = [f"| {key} | {val} |"]
    rows += ["|--|--|"]
    rows += [f"| {k} | {v} |" for k, v in d.items()]
    return "  \n".join(rows)


def log_config(config: DictConfig | ListConfig, writer):
    # Note sure how to type annotate the logger here
    """Log configuration to the logger, under 'text'."""

    flattened_config = flatten(config, reducer="dot", enumerate_types=(ListConfig, list))

    table = dict2mdtable(flattened_config, key="Hyperparameter", val="Value")
    # and finally, log a text scalar with a table
    writer.add_text("hyperparams", table)


def setup_wandb_logger(config: DictConfig | ListConfig, trainer: Engine, validator: Engine, optimizer: Optimizer):
    """Setup Weights and Biases logger."""
    # wandb_logger = WandBLogger(
    #     project=config.get("wandb_project", "ignition-project"),
    #     name=config.name,
    #     id=config.output_dir.split("/", 1)[-1],
    #     config=dict(config),
    #     save_dir=config.output_dir,
    #     notes=config.get("notes", ""),
    #     job_type="training"
    # )

    # wandb_logger.attach_output_handler(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED,
    #     tag="train",
    #     output_transform=lambda x: x,
        
    # )

    if not config.get("wandb_logging", False):
        return  # wandb logging not enabled

    if idist.get_rank() != 0:
        return  # only log from rank 0

    logger = setup_wandb_logging(
        trainer,
        optimizer,
        validator,
        config.log_every_iters,
        project=config.get("wandb_project", "ignition-project"),
        name=config.name,
        dir=config.output_dir,
        id=config.output_dir.split("/", 1)[-1],
        config=dict(config),
        notes=config.get("notes", ""),
        job_type="training"
    )

    printer.info("Weights and Biases logger initialized.")


def setup_tensorboard_writer(config: DictConfig | ListConfig, trainer, optimizers, evaluators):
    """Setup TensorBoard writer."""
    logger = common.setup_tb_logging(
        config.output_dir,
        trainer,
        optimizers,
        evaluators,
        config.log_every_iters,
    )

    # writer = SummaryWriter(
    #     log_dir=config.output_dir,
    # )
    writer = logger.writer

    printer.info("TensorBoard writer initialized at %s", writer.logdir)

    log_config(config, writer)

    return writer


def instantiate_handler(handler_config: DictConfig | ListConfig, **kwargs) -> Any:
    """Instantiate a handler from the configuration."""
    handler_requires_arg = handler_config.pop("_requires_", None)

    if handler_requires_arg is not None:
        if not isinstance(handler_requires_arg, ListConfig | list):
            handler_requires_arg = [handler_requires_arg]

        # If the handler requires some parameters, instantiate it with them
        required_args = {arg: kwargs[arg] for arg in handler_requires_arg}
        return instantiate(handler_config, _convert_="all", **required_args)
        # _convert_="all" is needed here to ensure that the required args are all passed as their inherent types.
        # This is needed because ignite's tree_apply function does not handle dictconfig objects as dicts..
    else:
        # Otherwise, instantiate it without any parameters
        return instantiate(handler_config)


def setup_train_handlers(
    config: dict,
    model: nn.Module,
    optimizer: Optimizer,
    writer: SummaryWriter,
    trainer: Engine,
    validator: Engine,
    lr_scheduler: LrScheduler | None = None,
):

    instantiate_kwargs = {
        "lr_scheduler": lr_scheduler,
        "validator": validator,
        "summary_writer": writer,
        "save_dir": os.path.join(config.output_dir, "checkpoints/train/"),
        "save_dict": {
            "model": model,
            "optimizer": optimizer,
            "trainer": trainer,
            # "validator": validator,
            "lr_scheduler": lr_scheduler,
        },
        # 'global_epoch_transform': get_epoch_function(trainer)  # not neded here, since we will use the trainer
    }

    # TODO: Which of these shoold run only on rank 0?
    for handler_config in config.handlers.get("train", []):
        handler = instantiate_handler(handler_config, **instantiate_kwargs)
        handler.attach(trainer)
        # Does the handler stay in scope from here?

    if config.get("resume") is not None:
        resume_path = config.resume_from_checkpoint
        instantiate_kwargs["load_path"] = resume_path
        instantiate_kwargs["load_dict"] = instantiate_kwargs["save_dict"]

        handler = instantiate_handler(config.handlers.checkpoint_loader, **instantiate_kwargs)
        printer.info("Resuming from checkpoint: %s", resume_path)
        handler(trainer)
        # Needs to run only once, at beginning


def setup_validation_handlers(
    config: dict,
    model: nn.Module,
    trainer: Engine,
    writer: SummaryWriter,
    validator: Engine,
):

    instantiate_kwargs = {
        "trainer": trainer,
        "summary_writer": writer,
        "save_dir": os.path.join(config.output_dir, "checkpoints/val"),
        "save_dict": {
            "model": model,
        },
        "global_epoch_transform": get_epoch_function(trainer),
    }

    for handler_config in config.handlers.get("validation", []):
        handler = instantiate_handler(handler_config, **instantiate_kwargs)
        handler.attach(validator)
