from torch.nn import Module
from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric

from ignite.engine import Engine, Events
from typing import Dict, Any, Optional, Union, Callable

class LossMetric:
    """This metric class only gives the epoch loss."""
    def __init__(self, loss_fn: Callable, model: Module):
        self.loss_fn = loss_fn
        self.output_transform = model.get_train_values_output_transform()

        self.metrics = {
            "epoch_loss": Loss(self.loss_fn, output_transform=self.output_transform)
        }

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def attach(self, engine: Engine, name: str = "epoch_loss") -> None:
        """Attach the metric to the engine."""
        engine.add_event_handler(Events.EPOCH_COMPLETED, lambda _: engine.state.metrics.update({name: self.compute(engine)}))