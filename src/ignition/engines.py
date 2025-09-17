import logging
from typing import Any, Dict, Union
from warnings import warn

import ignite.distributed as idist
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from ignite.engine import DeterministicEngine, Engine, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Metric
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from torch.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer

from ignition.datasets import PairedDataset, IgnitionDataset
from ignition.models import IgnitionModel
from ignition.utils import split_dict_at_index

logger = logging.getLogger(__name__)


def instantiate_post_transforms(pt_config: DictConfig, **kwargs):
    # TODO: Add typing for output (monai.Transform or torch.Transform?)
    # typically, the config should define a compose
    """
    Instantiate a post-transform based on the configuration.

    Args:
        pt_config (DictConfig): Configuration for the post-transform.
        **kwargs: Additional arguments to pass to the post-transform constructor.

    Returns:
        The instantiated post-transform.
    
    """
    if pt_config._target_ == "monai.transforms.Compose":
        # If it's a compose, instantiate each transform in the list
        transforms = []

        for transform_cfg in pt_config.transforms:
            required_args = transform_cfg.pop("_requires_", None)
            if required_args is not None:
                if not isinstance(required_args, ListConfig | list):
                    required_args = [required_args]
                required_args = {arg: kwargs[arg] for arg in required_args}
                transforms.append(instantiate(transform_cfg, **required_args))
            else:
                transforms.append(instantiate(transform_cfg))
        return instantiate(pt_config, transforms=transforms)
    else:
        # Otherwise, instantiate it directly
        required_args = pt_config.pop("_requires_", None)
        if required_args is not None:
            if not isinstance(required_args, ListConfig | list):
                required_args = [required_args]
            required_args = {arg: kwargs[arg] for arg in required_args}
            return instantiate(pt_config, **required_args)
        else:
            return instantiate(pt_config)


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
        warn("AMP may not be fully supported in Ignition.")
    else:
        scaler = None

    match config.engine_type:
        case "ignite":
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
                output_transform=model.get_train_output_transform(),
            )
            for name, metric in (metrics or {}).items():
                metric.attach(trainer, name)
        case "monai":
            # TODO: Add option to use slidingwindow inferer during training
            key_metric, other_metrics = split_dict_at_index(metrics, 1)

            trainer = SupervisedTrainer(
                device=device,
                max_epochs=config.max_epochs,
                train_data_loader=dataset.get_train_dataloader(),
                network=model,
                optimizer=optimizer,
                loss_function=loss_fn,
                non_blocking=True,
                postprocessing=instantiate(config.post_transforms.get("train")),
                key_train_metric=key_metric,
                additional_metrics=other_metrics,
            )

    return trainer


def setup_evaluator(
    config: Any,
    model: IgnitionModel,
    metrics: Dict[str, Metric],
    device: Union[str, torch.device],
    dataset: PairedDataset | IgnitionDataset,
    name: str | None = None,
    output_dir: str | None = None,
) -> Engine:

    # TODO: Move metrics declaration into here?
    instantiate_kwargs = {
        "output_dir": output_dir,
    }

    match config.engine_type:
        case "ignite":
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
        case "monai":

            key_metric, other_metrics = split_dict_at_index(metrics, 1)
            # relies on the fact that dicts are ordered in Python 3.7+

            # in a paired dataset, we get the val dataloader
            # otherwise, we get the single dataloader
            # TODO: Make this configurable?
            dataloader = dataset.get_val_dataloader() if isinstance(dataset, PairedDataset) else dataset.get_dataloader()

            post_transforms = instantiate_post_transforms(
                config.post_transforms.get(name) if name is not None else config.post_transforms,
                **instantiate_kwargs,
            )

            evaluator = SupervisedEvaluator(
                device=device,
                val_data_loader=dataloader,
                network=model,
                inferer=instantiate(config.inferer) if config.get("inferer") else None,
                postprocessing=post_transforms,
                val_handlers=instantiate(config.engine_handlers) if config.get("engine_handlers", False) else None,
                non_blocking=True,
                key_val_metric=key_metric,
                additional_metrics=other_metrics,
            )

    return evaluator
