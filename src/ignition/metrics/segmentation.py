from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric
from ignite.engine import Engine, Events
from typing import Dict

from .base import IgnitionMetrics

class SegmentationMetrics(IgnitionMetrics):
    def __init__(self, config, metrics_name: str = 'eval'):
        self.cm_metric = ConfusionMatrix(num_classes=config.num_classes)

        self.metrics = {}
        metrics_config = config.metrics.get(metrics_name)
        if metrics_config is not None:
            if metrics_config.get("miou", False):
                self.metrics["mIoU"] = mIoU(self.cm_metric, ignore_index=metrics_config.background_index)
            if metrics_config.get("iou", False):
                self.metrics["IoU"] = IoU(self.cm_metric, ignore_index=metrics_config.background_index)

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def attach_to_evaluator(self, evaluator: Engine):
        for name, metric in self.metrics.items():
            metric.attach(evaluator, name)