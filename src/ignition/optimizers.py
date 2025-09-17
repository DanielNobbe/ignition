from torch import optim
from hydra.utils import instantiate

def setup_optimizer(parameters, config):
    """
    Setup the optimizer based on the configuration.
    """
    if config.optimizer.get("type") == "SGD":
        return optim.SGD(
            parameters,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
    elif config.optimizer.get('_target_', False):
        return instantiate(config.optimizer, params=parameters)
    else:
        raise ValueError(
            f"Optimizer type {config.optimizer.type} is not supported. It can be implemented in optimizers.py"
        )
