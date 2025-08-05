from .loss import LossMetric
from .segmentation import SegmentationMetrics

from typing import Dict, Callable, Any
from torch.nn import Module
from ignite.metrics import Metric

def setup_metrics(config, metrics_name: str = 'eval', loss_fn: Callable | None = None, model: Module | None = None) -> Dict[str, Metric]:
    """Setup metrics based on the configuration."""
    
    if config.metrics[metrics_name].type == 'SegmentationMetrics':
        metric = SegmentationMetrics(config, metrics_name)
    elif config.metrics[metrics_name].type == 'LossMetric':
        if loss_fn is None or model is None:
            raise ValueError("Loss function and model must be provided for LossMetric.")
        metric = LossMetric(loss_fn, model=model)
    else:
        raise ValueError(f"Metrics type {config.metrics['metrics_name'].type} is not supported. It can be implemented in the metrics directory.")
    
    return metric.get_metrics()