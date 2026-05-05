from hydra.utils import instantiate
from torch.optim.lr_scheduler import PolynomialLR, LinearLR, SequentialLR, _LRScheduler

from omegaconf import DictConfig, ListConfig

from torch.optim import Optimizer

from typing import Any


def instantiate_scheduler(config: DictConfig, optimizer: Optimizer, _recursive_: bool = True, **kwargs) -> Any:
    """Instantiate a handler from the configuration."""
    handler_requires_arg = config.pop("_requires_", None)

    if handler_requires_arg is not None:
        if not isinstance(handler_requires_arg, ListConfig | list):
            handler_requires_arg = [handler_requires_arg]

        # If the handler requires some parameters, instantiate it with them
        required_args = {arg: kwargs[arg] for arg in handler_requires_arg}
        return instantiate(config, optimizer=optimizer, _recursive_=_recursive_, _convert_="all", **required_args)
        # _convert_="all" is needed here to ensure that the required args are all passed as their inherent types.
        # This is needed because ignite's tree_apply function does not handle dictconfig objects as dicts..
    else:
        # Otherwise, instantiate it without any parameters
        return instantiate(config, optimizer=optimizer, _recursive_=_recursive_)



def setup_lr_scheduler(optimizer, config, dataset_length):
    """
    Setup the learning rate scheduler based on the configuration.
    """
    if config.lr_scheduler.get("_target_", False):
        if 'WarmupLRWrapper' in config.lr_scheduler._target_:
            # non-recursively instantiate the scheduler, so init can handle the base scheduler
            return instantiate_scheduler(config.lr_scheduler, optimizer=optimizer, _recursive_=False)
        # If the config is a full path to a class, instantiate it
        return instantiate_scheduler(config.lr_scheduler, optimizer=optimizer)
    elif config.lr_scheduler.type == "PolynomialLR":
        return PolynomialLR(optimizer, power=config.lr_scheduler.power, total_iters=config.max_epochs * dataset_length)
    else:
        raise ValueError(
            f"Learning rate scheduler type {config.lr_scheduler.type} is not supported. It can be implemented in the lr_schedulers directory."
        )


class WarmupLRWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: DictConfig,
        warmup_multiplier: float = 0.0,
        last_epoch: int = -1,
    ):
        if not (0.0 <= warmup_multiplier <= 1.0):
            raise ValueError("warmup_multiplier must be in 0..1 range")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        if warmup_multiplier == 0.0 and warmup_steps > 0:
            start_factor = 1.0 / warmup_steps
        elif warmup_multiplier > 0.0 and warmup_steps > 0:
            start_factor = warmup_multiplier
        else:
            raise ValueError("Invalid warmup configuration.")

        # Linear warmup from warmup_multiplier * base_lr -> 1.0 * base_lr
        warmup = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
            last_epoch=last_epoch,
        )

        # Instantiate your real scheduler from Hydra config
        base_scheduler: _LRScheduler = instantiate(base_scheduler, optimizer=optimizer)

        self.scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, base_scheduler],
            milestones=[warmup_steps],
            last_epoch=last_epoch,
        )

    def step(self, epoch=None):
        if epoch is not None:
            raise ValueError("Epoch-based stepping is not supported in WarmupLRWrapper. Please use step() without arguments.")
        return self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    @property
    def _last_lr(self):
        return self.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.scheduler.load_state_dict(state_dict)