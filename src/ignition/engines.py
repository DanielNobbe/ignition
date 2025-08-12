from typing import Any, Dict, Union, Callable

import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Metric
from torch.cuda.amp import autocast, GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler
from ignition.models import IgnitionModel
from ignition.datasets import PairedDataset
from ignition.utils import split_dict_at_index

from monai.engines import SupervisedTrainer, SupervisedEvaluator

from hydra.utils import instantiate

import logging

logger = logging.getLogger(__name__)

def setup_trainer(
    config: Any,
    model: IgnitionModel,
    optimizer: Optimizer,
    loss_fn: Module,
    device: Union[str, torch.device],
    dataset: PairedDataset,
    metrics: Dict[str, Metric] = None,
) -> Union[Engine, DeterministicEngine]:

    if config.use_amp:
        scaler = GradScaler(enabled=config.use_amp)
    else:
        scaler = None

    match config.engine_type:
        case 'ignite':
            trainer = create_supervised_trainer(
                model,
                optimizer,
                loss_fn,
                device=device,
                prepare_batch=dataset.get_prepare_batch(),
                deterministic=True,
                non_blocking=True,
                scaler=scaler,
                model_transform=model.get_model_transform(),
                output_transform=model.get_train_output_transform()
            )
            for name, metric in (metrics or {}).items():
                metric.attach(trainer, name)
        case 'monai':
            # TODO: Add option to use slidingwindow inferer during training
            key_metric, other_metrics = split_dict_at_index(metrics, 1)

            # NOTE: it is odd, the metrics entered here should be ignite metrics,
            # but the MONAI metrics do not inherit from ignite.metrics.Metric
            # --> right, this doesn't actually work

            trainer = SupervisedTrainer(
                device=device,
                max_epochs=config.max_epochs,
                train_data_loader=dataset.get_train_dataloader(),
                network=model,
                optimizer=optimizer,
                loss_function=loss_fn,
                # prepare_batch=dataset.get_prepare_batch(),
                non_blocking=True,
                # model_transform=model.get_model_transform(),
                # output_transform=model.get_train_output_transform()
                key_train_metric=key_metric,
                additional_metrics=other_metrics,
            )


    return trainer


def setup_evaluator(
    config: Any,
    model: IgnitionModel,
    metrics: Dict[str, Metric],
    device: Union[str, torch.device],
    dataset: PairedDataset,
) -> Engine:

# TODO: Move metrics declaration into here?

    match config.engine_type:
        case 'ignite':
            evaluator = create_supervised_evaluator(
                model,
                metrics=metrics,
                device=device,
                non_blocking=True,
                prepare_batch=dataset.get_prepare_batch(),
                output_transform=model.get_eval_output_transform(),
            )
            for name, metric in metrics.items():
                metric.attach(evaluator, name)
        case 'monai':

            key_metric, other_metrics = split_dict_at_index(metrics, 1)

            evaluator = SupervisedEvaluator(
                device=device,
                val_data_loader=dataset.get_val_dataloader(),
                network=model,
                inferer=instantiate(config.inferer) if config.get('inferer') else None,
                # prepare_batch=dataset.get_prepare_batch(),
                non_blocking=True,
                key_val_metric=key_metric,
                additional_metrics=other_metrics,
            )
    # TODO: maybe we need to use the postprocessing arg for the transform?



    return evaluator
