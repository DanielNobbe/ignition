import torch.nn as nn
from hydra.utils import instantiate


def setup_loss(config) -> nn.Module:
    """
    Setup the loss function based on the configuration.
    """
    match config.loss.type:
        case "CrossEntropyLoss":
            return nn.CrossEntropyLoss(
                ignore_index=config.loss.ignore_index, label_smoothing=config.loss.label_smoothing
            )
        case "monai":
            hy_config = config.loss.copy()
            hy_config.pop("type", None)  # Remove type to avoid conflicts with hydra
            return instantiate(hy_config)
        case _:
            raise ValueError(
                f"Loss type {config.loss.type} is not supported. It can be implemented in the losses directory or through losses/__init__.py."
            )
