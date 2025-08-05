from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric
from ignite.engine import Engine, Events
from typing import Dict

from abc import ABC, abstractmethod

class IgnitionMetrics(ABC):

    @abstractmethod
    def get_metrics(self) -> Dict[str, Metric]:
        """Return a dictionary of metrics."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def attach_to_evaluator(self, evaluator: Engine):
        """Attach metrics to the evaluator."""
        for name, metric in self.get_metrics().items():
            metric.attach(evaluator, name)