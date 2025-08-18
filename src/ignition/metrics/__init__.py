from .loss import LossMetric, MonaiLossMetric
from .segmentation import SegmentationMetrics

from typing import Dict, Callable, Any
from torch.nn import Module
from ignite.metrics import Metric
from ignite.engine import Engine
from ignite.handlers import global_step_from_engine

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

def instantiate_metric(
    metric_config: DictConfig, **kwargs
) -> Dict[str, Metric]:
    """
    Instantiate a metric based on the configuration.
    
    Args:
        metric_config (DictConfig): Configuration for the metric.
        **kwargs: Additional arguments to pass to the metric constructor.

    Returns:
        Dict[str, Metric]: Dictionary containing the instantiated metric.
    
    TODO: Ensure output is actually a dict with ignite metrics? 
    """
    metric_requires_arg = metric_config.pop("_requires_", None)

    if metric_requires_arg is not None:
        if not isinstance(metric_requires_arg, ListConfig | list):
            metric_requires_arg = [metric_requires_arg]

        # If the metric requires some parameters, instantiate it with them
        required_args = {arg: kwargs[arg] for arg in metric_requires_arg}
        return instantiate(metric_config, **required_args)
    else:
        # Otherwise, instantiate it without any parameters
        return instantiate(metric_config)
    

# def setup_metrics(
#     config: DictConfig,
#     metrics_name: str = 'val',
#     loss_fn: Callable | None = None,
#     model: Module | None = None
# ) -> Dict[str, Metric]:
#     """
#     Setup metrics based on the configuration.
    
#     Args:
#         config (DictConfig): Configuration object.
#         metrics_name (str): Name of the metrics configuration to use.
#         loss_fn (Callable, optional): Loss function for loss metrics.
#         model (Module, optional): Model for loss metrics.

#     Returns:
#         Dict[str, Metric]: Dictionary of configured metrics.
#     """
#     return setup_metrics(config, metrics_name, loss_fn, model)
# )


def setup_metrics(config, metrics_name: str = 'val', loss_fn: Callable | None = None, model: Module | None = None, trainer: Engine | None = None) -> Dict[str, Metric]:
    """Setup metrics based on the configuration."""

    instantiate_kwargs = {
        'loss_fn': loss_fn,
        'model': model,
        **({'global_step_transform': global_step_from_engine(trainer)} if trainer is not None else {})
    }

    metrics = {}
    for metric_config in config.metrics.get(metrics_name, {}):
        metric_config = metric_config.copy()
        metric_key = metric_config.pop('_key_')
        metrics.update(({metric_key: instantiate_metric(metric_config, **instantiate_kwargs)}))

    first_key = next(iter(metrics))
    if not config.metrics.get(f"{metrics_name}_key_metric_name") == first_key:
        raise ValueError(
            f"First metric key '{first_key}' does not match expected key '{config.metrics.get(f'{metrics_name}_key_metric_name')}'."
        )

    return metrics
    
    if config.metrics[metrics_name].type == 'SegmentationMetrics':
        metric = SegmentationMetrics(config, metrics_name)
    elif config.metrics[metrics_name].type == 'LossMetric':
        if loss_fn is None or model is None:
            raise ValueError("Loss function and model must be provided for LossMetric.")
        if config.model.type == 'monai':
            metric = MonaiLossMetric(loss_fn, model=model)
        else:
            metric = LossMetric(loss_fn, model=model)
    else:
        raise ValueError(f"Metrics type {config.metrics['metrics_name'].type} is not supported. It can be implemented in the metrics directory.")
    
    return metric.get_metrics()