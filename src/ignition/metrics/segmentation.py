from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric
from ignite.engine import Engine, Events
from typing import Dict

class SegmentationMetrics:
    def __init__(self, config, metrics_name: str = 'eval'):
        self.cm_metric = ConfusionMatrix(num_classes=config.num_classes)

        self.metrics = {}
        metrics_config = config.metrics.get(metrics_name)
        if metrics_config is not None:
            if metrics_config.get("IoU", True):
                self.metrics["IoU"] = IoU(self.cm_metric)
            if metrics_config.get("mIoU_bg", True):
                self.metrics["mIoU_bg"] = mIoU(self.cm_metric)

    def get_metrics(self) -> Dict[str, Metric]:
        return self.metrics

    def attach_to_evaluator(self, evaluator: Engine):
        for name, metric in self.metrics.items():
            metric.attach(evaluator, name)