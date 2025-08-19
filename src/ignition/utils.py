import logging
import numbers
from datetime import datetime
from logging import Logger, getLogger
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import ignite.distributed as idist
import torch
from ignite.engine import Engine
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from omegaconf import OmegaConf

printer = getLogger(__name__)


def setup_config(config):
    OmegaConf.set_struct(config, False)

    config.backend = config.get("backend", None)

    return config


def split_dict_at_index(d: dict, index: int) -> tuple[dict, dict]:
    """Split a dictionary into two parts at the given index."""
    if not isinstance(d, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(index, numbers.Integral):
        raise TypeError("Index must be an integer.")

    keys = list(d.keys())
    if index < 0 or index > len(keys):
        raise IndexError("Index out of range.")

    first_part = {k: d[k] for k in keys[:index]}
    second_part = {k: d[k] for k in keys[index:]}

    return first_part, second_part


def log_metrics(engine: Engine, tag: str) -> None:
    """Log `engine.state.metrics` with given `engine` and `tag`.

    Parameters
    ----------
    engine
        instance of `Engine` which metrics to log.
    tag
        a string to add at the start of output.
    """
    metrics_format = "{0} [{1},{2}]: {3}".format(tag, engine.state.epoch, engine.state.iteration, engine.state.metrics)
    engine.logger.info(metrics_format)


def resume_from(
    to_load: Mapping,
    checkpoint_fp: Union[str, Path],
    logger: Logger,
    strict: bool = True,
    model_dir: Optional[str] = None,
) -> None:
    """Loads state dict from a checkpoint file to resume the training.

    Parameters
    ----------
    to_load
        a dictionary with objects, e.g. {“model”: model, “optimizer”: optimizer, ...}
    checkpoint_fp
        path to the checkpoint file
    logger
        to log info about resuming from a checkpoint
    strict
        whether to strictly enforce that the keys in `state_dict` match the keys
        returned by this module’s `state_dict()` function. Default: True
    model_dir
        directory in which to save the object
    """
    if isinstance(checkpoint_fp, str) and checkpoint_fp.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_fp,
            model_dir=model_dir,
            map_location="cpu",
            check_hash=True,
        )
    else:
        if isinstance(checkpoint_fp, str):
            checkpoint_fp = Path(checkpoint_fp)

        if not checkpoint_fp.exists():
            raise FileNotFoundError(f"Given {str(checkpoint_fp)} does not exist.")
        checkpoint = torch.load(checkpoint_fp, map_location="cpu")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_fp)


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    output_dir = config.output_dir
    if rank == 0:
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{config.output_prefix}_{now}-backend-{config.backend}"
        path = Path(config.output_dir, name)
        path.mkdir(parents=True, exist_ok=True)
        output_dir = path.as_posix()
    return Path(idist.broadcast(output_dir, src=0))


def save_config(config, output_dir):
    """Save configuration to config-lock.yaml for result reproducibility."""
    with open(f"{output_dir}/config-lock.yaml", "w") as f:
        OmegaConf.save(config, f)


def setup_logging(config: Any) -> Logger:
    """Setup logger with `ignite.utils.setup_logger()`.

    Parameters
    ----------
    config
        config object. config has to contain `verbose` and `output_dir` attribute.

    Returns
    -------
    logger
        an instance of `Logger`
    """
    green = "\033[32m"
    reset = "\033[0m"
    logger = setup_logger(
        name=f"{green}[ignite]{reset}",
        level=logging.DEBUG if config.debug else logging.INFO,
        filepath=config.output_dir / "training-info.log",
    )
    return logger


def get_key_metric_value(engine):
    return engine.state.metrics.get(engine.key_metric_name, None)


def get_epoch_function(engine: Engine) -> int:
    """Get the current epoch number from the engine state."""

    def get_epoch(*args, **kwargs):
        return engine.state.epoch

    return get_epoch
