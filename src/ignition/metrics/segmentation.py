from ignite.metrics import ConfusionMatrix, IoU, mIoU, Loss, Metric
from ignite.engine import Engine, Events
from typing import Dict

from .base import IgnitionMetrics

import torch

def monai_model_transform(model_output):
    # TODO: Move to ignition.utils
    """Transform for MONAI models to get the output."""
    if isinstance(model_output, list):
        assert model_output[0]['label'].shape[0] == 1, "Expected batch size of 1 for MONAI model output per list item."
        assert len(model_output[0]['label'].shape) == len(model_output[0]['pred'].shape), "Expected labels to have the same number of dimensions as the predictions."  #--> actually not, the labels are not one-hot encoded but allow for channels or a batch dimension, while the predictions are one-hot encoded. So they should have the same number of dimensions.

        # Concatenate predictions and labels from the list of dictionaries
        y_preds = [item['pred'].unsqueeze(0) for item in model_output]
        ys = [item['label'] for item in model_output]

        y_pred = torch.cat(y_preds, dim=0)
        y = torch.cat(ys, dim=0).to(torch.long)
    else:
        raise ValueError("Expected model output to be a list of dictionaries with 'pred' and 'label' keys.")
    
    return y_pred, y



class SegmentationMetrics(IgnitionMetrics):
    def __init__(self, config, metrics_name: str = 'eval'):
        self.cm_metric = ConfusionMatrix(
            num_classes=config.num_classes,
            output_transform=monai_model_transform if config.model.type == 'monai' else lambda x: x,
            )

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