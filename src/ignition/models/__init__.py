from .abstract import IgnitionModel
from .torchvision import TorchVisionSegmentationModel

def setup_model(config):
    if config.model.type == "torchvision":
        return TorchVisionSegmentationModel(config)
    else:
        raise ValueError(f"Model type {config.model.type} is not supported.")