from hydra.utils import instantiate
from torch.optim.lr_scheduler import PolynomialLR

from omegaconf import DictConfig, ListConfig

from torch.optim import Optimizer

from typing import Any


def instantiate_scheduler(config: DictConfig, optimizer: Optimizer, **kwargs) -> Any:
    """Instantiate a handler from the configuration."""
    handler_requires_arg = config.pop("_requires_", None)

    if handler_requires_arg is not None:
        if not isinstance(handler_requires_arg, ListConfig | list):
            handler_requires_arg = [handler_requires_arg]

        # If the handler requires some parameters, instantiate it with them
        required_args = {arg: kwargs[arg] for arg in handler_requires_arg}
        return instantiate(config, optimizer=optimizer, _convert_="all", **required_args)
        # _convert_="all" is needed here to ensure that the required args are all passed as their inherent types.
        # This is needed because ignite's tree_apply function does not handle dictconfig objects as dicts..
    else:
        # Otherwise, instantiate it without any parameters
        return instantiate(config, optimizer=optimizer)



def setup_lr_scheduler(optimizer, config, dataset_length):
    """
    Setup the learning rate scheduler based on the configuration.
    """
    if config.lr_scheduler.get("_target_", False):
        # If the config is a full path to a class, instantiate it
        return instantiate_scheduler(config.lr_scheduler, optimizer=optimizer)
    elif config.lr_scheduler.type == "PolynomialLR":
        return PolynomialLR(optimizer, power=config.lr_scheduler.power, total_iters=config.max_epochs * dataset_length)
    else:
        raise ValueError(
            f"Learning rate scheduler type {config.lr_scheduler.type} is not supported. It can be implemented in the lr_schedulers directory."
        )
