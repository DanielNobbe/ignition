from monai.handlers import (
    LrScheduleHandler,
    ValidationHandler,
    StatsHandler,
    TensorBoardStatsHandler,
    CheckpointSaver,
    EarlyStopHandler,
    TensorBoardImageHandler,
    from_engine
)
from ignite.handlers import ProgressBar, global_step_from_engine

from ignite.engine import Engine
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler as LrScheduler
import os

from ignite.contrib.engines import common
from ignite.contrib.handlers.tensorboard_logger import OutputHandler

from typing import Any, Iterable
from flatten_dict import flatten
import numbers

from omegaconf import DictConfig, ListConfig

from hydra.utils import instantiate

from ignition.utils import get_epoch_function

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
    lr_scheduler: LrScheduleHandler | None = None,
):
    """Setup all handlers for the trainer and validator."""
    writer = setup_tensorboard_writer(config, trainer, optimizer, validator)

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


def dict2mdtable(d, key='Name', val='Value'):
    rows = [f'| {key} | {val} |']
    rows += ['|--|--|']
    rows += [f'| {k} | {v} |' for k, v in d.items()]
    return "  \n".join(rows)



def log_config(config: DictConfig | ListConfig, writer):
    # Note sure how to type annotate the logger here
    """Log configuration to the logger, under 'text'.
    """

    flattened_config = flatten(config, reducer="dot", enumerate_types=(ListConfig, list))

    table = dict2mdtable(flattened_config, key="Hyperparameter", val="Value")
    # and finally, log a text scalar with a table
    writer.add_text(
        "hyperparams",
        table
    )


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

    printer.info("TensorBoard writer initialized at %s", writer.log_dir)

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
        return instantiate(handler_config, **required_args)
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
        'lr_scheduler': lr_scheduler,
        'validator': validator,
        'summary_writer': writer,
        'save_dir': os.path.join(writer.get_logdir(), "checkpoints"),
        'save_dict': {
            "model": model,
            "optimizer": optimizer,
            "trainer": trainer,
            # "validator": validator,
            "lr_scheduler": lr_scheduler,
        },
        # 'global_epoch_transform': get_epoch_function(trainer)  # not neded here, since we will use the trainer
    }

    for handler_config in config.handlers.get('train', []):
        handler = instantiate_handler(handler_config, **instantiate_kwargs)
        handler.attach(trainer)
        # Does the handler stay in scope from here?

    # # TODO: Which of these shoold run only on rank 0?
    # lr_schedule_handler = LrScheduleHandler(
    #     lr_scheduler=lr_scheduler,
    #     print_lr=True,
    #     epoch_level=False
    #     )  # no need to specify step_transform, since most lr schedulers only rely on the parameter gradients
    
    # lr_schedule_handler.attach(trainer)

    # validation_handler = ValidationHandler(
    #     interval=5,  # every 5 epochs
    #     validator=validator,
    #     epoch_level=True,  # validation is done at the end of each epoch
    #     exec_at_start=True,  # run validation at the start of training
    # )
    # validation_handler.attach(trainer)

    # stats_handler = StatsHandler(
    #     iteration_log=True,
    #     epoch_log=False,
    #     tag_name="train loss"
    # )
    # # use default iteration_print_logger - it prints the loss
    # # not sure if the output_transform is correct by default

    # stats_handler.attach(trainer)

    # # This should just add an additional logging for the loss?
    # tensorboard_stats_handler = TensorBoardStatsHandler(
    #     summary_writer=writer,
    #     iteration_log=True,
    #     epoch_log=True,
    #     tag_name="train loss"
    # )
    # tensorboard_stats_handler.attach(trainer)

    # # checkpoint saver
    # # TODO: Make folder with logs + checkpoints
    # # TODO: Make this the first handler to save on exception
    # checkpoint_saver = CheckpointSaver(
    #     save_dir=os.path.join(writer.get_logdir(), "checkpoints"),
    #     save_dict={
    #         "model": model,
    #         "optimizer": optimizer,
    #         "trainer": trainer,
    #         # "validator": validator,
    #         "lr_scheduler": lr_scheduler,
    #     },
    #     save_final=True,
    #     final_filename="final_checkpoint.pth",
    #     save_key_metric=False,
    #     epoch_level=True,
    #     save_interval=1,  # save every epoch
    #     n_saved=5,  # keep last 5 checkpoints
    # )
    # checkpoint_saver.attach(trainer)

    # pbar = ProgressBar(persist=True, desc="Training Progress")
    # pbar.attach(trainer)


def setup_validation_handlers(
    config: dict,
    model: nn.Module,
    trainer: Engine,
    writer: SummaryWriter,
    validator: Engine,
):
    
    instantiate_kwargs = {
        'trainer': trainer,
        'summary_writer': writer,
        'save_dir': os.path.join(writer.get_logdir(), "checkpoints"),
        'save_dict': {
            "model": model,
        },
        # **({'global_step_transform': global_step_from_engine(trainer)} if trainer is not None else {})
        'global_epoch_transform': get_epoch_function(trainer),
    }

    for handler_config in config.handlers.get('validation', []):
        handler = instantiate_handler(handler_config, **instantiate_kwargs)
        handler.attach(validator)

    # # this is optional! We should already use the checkpointsaver to save the best model
    # early_stop_handler = EarlyStopHandler(
    #     patience=5,
    #     score_function=lambda engine: engine.state.metrics[engine.state.key_metric_name],
    #     trainer=trainer,
    #     min_delta=0.0001,
    #     cumulative_delta=False,
    #     epoch_level=True,
    # )
    # early_stop_handler.attach(validator)


    checkpoint_saver = CheckpointSaver(
        save_dir=os.path.join(writer.get_logdir(), "checkpoints"),
        save_dict={
            "model": model,
        },
        file_prefix="best_model",
        save_final=False,
        save_key_metric=True,
        epoch_level=True,
        save_interval=1,  # save every epoch
        key_metric_n_saved=1,  # keep only the best model
        key_metric_greater_or_equal=True,  # if two have same metric, save the last one
    )
    # checkpoint_saver.attach(validator)

    # stats_handler = StatsHandler(
    #     iteration_log=False,
    #     epoch_log=True,
    #     tag_name="validation"
    # )  # TODO: Check if this prints all validation metrics?
    # stats_handler.attach(validator)

    # tensorboard_stats_handler = TensorBoardStatsHandler(
    #     summary_writer=writer,
    #     iteration_log=False,
    #     epoch_log=True,
    #     tag_name="validation",
    #     # state_attributes=["metrics"]
    # )  # TODO: Check if this saves all validation metrics
    # tensorboard_stats_handler.attach(validator)

    # tensorboard_image_handler = TensorBoardImageHandler(
    #     summary_writer=writer,
    #     epoch_level=True,
    #     interval=1,
    #     index=0,
    #     batch_transform=from_engine(['image', 'label']),
    #     output_transform=from_engine(['pred']),
    # )
    # tensorboard_image_handler.attach(validator)