import torch.nn as nn


def setup_loss(config) -> nn.Module:
    """
    Setup the loss function based on the configuration.
    """
    if config.loss.type == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(
            ignore_index=config.loss.ignore_index,
            label_smoothing=config.loss.label_smoothing
        )
    else:
        raise ValueError(f"Loss type {config.loss.type} is not supported. It can be implemented in the losses directory or through losses/__init__.py.")