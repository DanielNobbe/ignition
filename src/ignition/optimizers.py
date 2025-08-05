from torch import optim


def setup_optimizer(parameters, config):
    """
    Setup the optimizer based on the configuration.
    """
    if config.optimizer.type == "SGD":
        return optim.SGD(
            parameters,
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=config.optimizer.nesterov,
        )
    else:
        raise ValueError(f"Optimizer type {config.optimizer.type} is not supported. It can be implemented in optimizers.py")