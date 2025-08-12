from .base import IgnitionModel
from .torchvision import TorchVisionSegmentationModel

from hydra.utils import instantiate

def setup_model(config):
    match config.model.type:
        case "torchvision":
            return TorchVisionSegmentationModel(config)
        case "monai":
            hy_config = config.model.copy()
            hy_config.pop('type', None)  # Remove type to avoid conflicts with hydra
            return instantiate(hy_config)
        case _:
            raise ValueError(f"Model type {config.model.type} is not supported. It can be implemented in the models directory.")