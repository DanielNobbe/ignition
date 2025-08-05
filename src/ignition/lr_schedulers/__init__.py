from torch.optim.lr_scheduler import PolynomialLR
from functools import partial


def setup_lr_scheduler(optimizer, config, dataset_length):
    """
    Setup the learning rate scheduler based on the configuration.
    """
    if config.lr_scheduler.type == "PolynomialLR":
        return PolynomialLR(
            optimizer,
            power=config.lr_scheduler.power,
            total_iters=config.max_epochs * dataset_length
        )
    else:
        raise ValueError(f"Learning rate scheduler type {config.lr_scheduler.type} is not supported. It can be implemented in the lr_schedulers directory.")