from .logger import WandBLogger

from typing import Any, Callable, cast, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ignite.engine import Engine, Events
from ignite.handlers import global_step_from_engine
from ignite.handlers.base_logger import BaseLogger


def setup_logging(
    logger: BaseLogger,
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer], Dict[None, Optimizer]]],
    evaluators: Optional[Union[Engine, Dict[str, Engine]]],
    log_every_iters: int,
) -> None:
    if optimizers is not None:
        if not isinstance(optimizers, (Optimizer, Mapping)):
            raise TypeError("Argument optimizers should be either a single optimizer or a dictionary or optimizers")

    if evaluators is not None:
        if not isinstance(evaluators, (Engine, Mapping)):
            raise TypeError("Argument evaluators should be either a single engine or a dictionary or engines")

    if log_every_iters is None:
        log_every_iters = 1

    logger.attach_output_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=log_every_iters), tag="training", metric_names="all"
    )

    if optimizers is not None:
        # Log optimizer parameters
        if isinstance(optimizers, Optimizer):
            optimizers = {None: optimizers}

        for k, optimizer in optimizers.items():
            logger.attach_opt_params_handler(
                trainer, Events.ITERATION_STARTED(every=log_every_iters), optimizer, param_name="lr", tag=k
            )

    if evaluators is not None:
        # Log evaluation metrics
        if isinstance(evaluators, Engine):
            evaluators = {"validation": evaluators}

        event_name = Events.ITERATION_COMPLETED if isinstance(logger, WandBLogger) else None
        print(f"Setting up logging for evaluators with event name: {event_name} for logger type {type(logger)}")
        gst = global_step_from_engine(trainer, custom_event_name=event_name)
        for k, evaluator in evaluators.items():
            logger.attach_output_handler(
                evaluator, event_name=Events.COMPLETED, tag=k, metric_names="all", global_step_transform=gst
            )



def setup_wandb_logging(
    trainer: Engine,
    optimizers: Optional[Union[Optimizer, Dict[str, Optimizer]]] = None,
    evaluators: Optional[Union[Engine, Dict[str, Engine]]] = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> WandBLogger:
    """Method to setup WandB logging on trainer and a list of evaluators. Logged metrics are:

        - Training metrics, e.g. running average loss values
        - Learning rate(s)
        - Evaluation metrics

    Args:
        trainer: trainer engine
        optimizers: single or dictionary of
            torch optimizers. If a dictionary, keys are used as tags arguments for logging.
        evaluators: single or dictionary of evaluators. If a dictionary,
            keys are used as tags arguments for logging.
        log_every_iters: interval for loggers attached to iteration events. To log every iteration,
            value can be set to 1 or None.
        kwargs: optional keyword args to be passed to construct the logger.

    Returns:
        :class:`~ignite.handlers.wandb_logger.WandBLogger`
    """
    logger = WandBLogger(**kwargs)
    setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger
