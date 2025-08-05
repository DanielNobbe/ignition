from typing import Any, Dict, Union

import ignite.distributed as idist
import torch
from .data import prepare_image_mask
from ignite.engine import DeterministicEngine, Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Metric
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler
from .utils import model_output_transform, model_eval_output_transform, model_train_output_transform

import logging

logger = logging.getLogger(__name__)

def setup_trainer(
    config: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    device: Union[str, torch.device],
    metrics: Dict[str, Metric] = None,
) -> Union[Engine, DeterministicEngine]:
    prepare_batch = prepare_image_mask

    if config.use_amp:
        scaler = GradScaler(enabled=config.use_amp)
    else:
        scaler = None

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        prepare_batch=prepare_batch,
        deterministic=True,
        non_blocking=True,
        scaler=scaler,
        model_transform=model_output_transform,
        output_transform=model_train_output_transform
    )

    for name, metric in (metrics or {}).items():
        metric.attach(trainer, name)

    return trainer


def setup_evaluator(
    config: Any,
    model: Module,
    metrics: Dict[str, Metric],
    device: Union[str, torch.device],
) -> Engine:
    prepare_batch = prepare_image_mask

    evaluator = create_supervised_evaluator(
        model,
        metrics=metrics,
        device=device,
        non_blocking=True,
        prepare_batch=prepare_batch,
        output_transform=model_eval_output_transform,
    )

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator
