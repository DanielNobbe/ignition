from typing import Any, Callable, Dict

from hydra.utils import instantiate
from ignite.engine import Engine
from ignite.handlers import global_step_from_engine
from ignite.metrics import Metric
from omegaconf import DictConfig, ListConfig
from torch.nn import Module


def instantiate_metric(metric_config: DictConfig, **kwargs) -> Dict[str, Metric]:
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


def setup_metrics(
    config,
    metrics_name: str | None = None,
    loss_fn: Callable | None = None,
    model: Module | None = None,
    trainer: Engine | None = None,
) -> Dict[str, Metric]:
    """Setup metrics based on the configuration."""

    instantiate_kwargs = {
        "loss_fn": loss_fn,
        "model": model,
        **({"global_step_transform": global_step_from_engine(trainer)} if trainer is not None else {}),
    }

    metrics = {}
    metrics_key = metrics_name if metrics_name is not None else "list"
    metrics_config_list = config.metrics.get(metrics_key, [])
    if not metrics_config_list:
        raise ValueError(f"No metrics found in config for '{metrics_name}'." if metrics_name else "No metrics found in config.")

    for metric_config in metrics_config_list:
        metric_config = metric_config.copy()
        metric_key = metric_config.pop("_key_")
        metrics.update(({metric_key: instantiate_metric(metric_config, **instantiate_kwargs)}))

    first_key = next(iter(metrics))
    key_metric_name = f"{metrics_name}_key_metric_name" if metrics_name is not None else "key_metric_name"
    if not config.metrics.get(key_metric_name) == first_key:
        raise ValueError(
            f"First metric key '{first_key}' does not match expected key '{config.metrics.get(f'{metrics_key}_key_metric_name')}'."
        )

    return metrics
