from torch.nn import Module
from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric

from ignite.engine import Engine, Events
from typing import Dict, Any, Optional, Union, Callable

from .base import IgnitionMetrics

from ignition.models import IgnitionModel

from monai.metrics import LossMetric as MonaiLoss

from monai.handlers import IgniteMetricHandler, from_engine

class LossMetric(IgnitionMetrics):
    """This metric class only gives the epoch loss."""
    def __init__(self, loss_fn: Callable, model: Module):
        self.loss_fn = loss_fn

        if isinstance(model, IgnitionModel):
            self.output_transform = model.get_train_values_output_transform()
        else:
            # in this case we may have a generic model that we don't need to wrap
            self.output_transform = lambda x: x  

        self.metrics = {
            "epoch_loss": Loss(self.loss_fn, output_transform=self.output_transform)
        }

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

class MonaiLossMetric(IgnitionMetrics):
    """This metric class only gives the epoch loss for MONAI models."""
    def __init__(self, loss_fn: Callable, model: Module):
        self.loss_fn = loss_fn
        self.output_transform = lambda x: x  # No specific output transform for MONAI

        self.metrics = {
            "epoch_loss": IgniteMetricHandler(loss_fn=self.loss_fn, output_transform=from_engine(("pred", "label"))) #MonaiLoss(self.loss_fn)
        }

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics
