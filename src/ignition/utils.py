import sys

import logging
import numbers
from datetime import datetime
from logging import Logger, getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import ignite.distributed as idist
import torch
from ignite.engine import Engine
from ignite.handlers import Checkpoint
from ignite.utils import setup_logger
from monai.transforms import Transform
from monai.handlers import from_engine
from monai.config import KeysCollection
from omegaconf import OmegaConf, DictConfig

import re

printer = getLogger(__name__)

def handle_num_workers(config):
    """Set num_workers in config based on the number of available CPUs."""
    if config.get("num_workers") is None:
        num_cores = cpu_count()
        if num_cores is None:
            num_cores = 1
        num_workers = num_cores // idist.get_world_size()
        if num_workers == 0:
            num_workers = 1
        config.num_workers = num_workers
        printer.info(f"Setting num_workers in config based on available CPUs ({num_workers}).")


def setup_config(config):
    OmegaConf.set_struct(config, False)

    config.backend = config.get("backend", None)

    handle_num_workers(config)

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
    strip_compiled: bool = False,
    strip_ddp: bool = False,
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

    if strip_compiled:
        if "model" in to_load:
            checkpoint["model"] = state_dict_strip_compiled(checkpoint["model"])
            logger.info("Stripped '_orig_mod.' prefix from model state dict keys (from torch.compile).")
    if strip_ddp:
        if "model" in to_load:
            checkpoint["model"] = state_dict_strip_ddp(checkpoint["model"])
            logger.info("Stripped 'module.' prefix from model state dict keys (from DDP).")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint, strict=strict)
    logger.info("Successfully resumed from a checkpoint: %s", checkpoint_fp)


def setup_output_dir(config: Any, rank: int) -> Path:
    """Create output folder."""
    output_dir = config.output_dir
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{config.output_prefix}_{now}-backend-{config.backend}"
    path = Path(config.output_dir, name)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory at: {path}", flush=True)
    output_dir = path.as_posix()
    return output_dir


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
        filepath=config.output_dir + "/training-info.log",
        reset=True
    )
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    return logger


def get_key_metric_value(engine):
    return engine.state.metrics.get(engine.key_metric_name, None)


def get_epoch_function(engine: Engine) -> int:
    """Get the current epoch number from the engine state."""

    def get_epoch(*args, **kwargs):
        return engine.state.epoch

    return get_epoch


def find_best_checkpoint(checkpoint_dir: Union[str, Path]):
    # very simple, under the checkpoint/val folder, there is a single file with the name
    # model_key_metric=xx.pt
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    
    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}.")
    
    # One of the checkpoints will have 'model_key_metric=' in its name
    # get all matches to make sure there's only one
    best_checkpoints = [ckpt for ckpt in checkpoints if "model_key_metric=" in ckpt.name]
    if len(best_checkpoints) > 1:
        raise ValueError(f"Multiple best checkpoints found in {checkpoint_dir}: {best_checkpoints}")
    if len(best_checkpoints) == 0:
        return find_last_checkpoint(checkpoint_dir)
    if len(best_checkpoints) == 1:
        return best_checkpoints[0]


def find_last_checkpoint(checkpoint_dir: Union[str, Path]) -> Path | None:
    """Find the last checkpoint file in the given directory."""
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        return None

    # the checkpoints have names with the epoch number, so we could get the last one by 
    # sorting alphabetically.
    # This does not take into account the 'final_model.pth' checkpoint though, which is always okay to take

    final_model_matches = [ckpt for ckpt in checkpoints if "final_model" in ckpt.name]
    if len(final_model_matches) > 1:
        raise ValueError(f"Multiple final_model checkpoints found in {checkpoint_dir}: {final_model_matches}")
    if len(final_model_matches) == 1:
        return final_model_matches[0]
    
    # if there is no final model, use the model with the highest epoch number

    # Sort by name, the highest number should be first.
    # this is problematic though because 999 would come before 1000
    def extract_epoch(ckpt: Path) -> int:
        match = re.search(r"epoch=(\d+)", ckpt.name)
        if match:
            return int(match.group(1))
        else:
            return -1  # if no epoch found, put it at the start
    checkpoints.sort(key=extract_epoch, reverse=True)
    return checkpoints[0] if checkpoints else None


def get_model_config(model_dir, config_name="config-lock.yaml"):
    """Load model configuration from a specified directory."""
    config_file = Path(model_dir) / config_name
    if not config_file.exists():
        raise FileNotFoundError(f"Could not find {config_name} in directory {model_dir}")
    
    model_config = OmegaConf.load(config_file)
    return model_config


def load_pretrained_weights(config: DictConfig, model: torch.nn.Module, logger: Logger):
    """Load pretrained weights, if specified in the config."""
    if config.get("pretrained"):
        weights_file = config.pretrained.get("weights_file", None)
        if weights_file is None:
            raise ValueError("If 'pretrained' is specified in the config, 'weights_file' must be provided.")
        if not Path(weights_file).exists():
            raise FileNotFoundError(f"Pretrained weights file {weights_file} does not exist.")
        
        logger.info(f"Loading pretrained weights from {weights_file}")
        checkpoint = torch.load(weights_file, map_location="cpu")

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=config.pretrained.get("strict", True))  # TODO: Check the structure of the state dict, if anything gets loaded correctly
        logger.info("Successfully loaded pretrained weights.")


def load_checkpoint_for_evaluation(config: DictConfig, model_dir: str, model: torch.nn.Module, logger: Logger):
    """Load model weights from the best checkpoint for evaluation.

    Parameters
    ----------
    config
        configuration object, must contain `train_dir` field pointing to training output directory.
    model
        model to load weights into
    logger
        logger to log info about loading the checkpoint

    Returns
    -------
    model
        model with loaded weights
    """
    
    last_checkpoint = find_best_checkpoint(Path(model_dir) / "checkpoints/val")
    if last_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")
    
    logger.info(f"Loading model weights from checkpoint: {last_checkpoint}")

    strip_compiled = not config.get("compile", False)  # if we're not evaluating with compile, we need to strip the compiled prefixes
    strip_ddp = not config.get("distributed", False)  # if we're not evaluating with distributed, we need to strip the DDP prefixes
    
    to_load = {"model": model}
    resume_from(to_load, last_checkpoint, logger, strict=True, strip_compiled=strip_compiled, strip_ddp=strip_ddp)
    
    return model


def resume_from_log(resume_path: str, config_name: str = "config-lock.yaml") -> DictConfig:
    """Resume configuration from a checkpoint file."""
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"Resume path does not exist: {resume_path}")

    resume_yaml = Path(resume_path) / config_name
    if not resume_yaml.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {resume_yaml}")
    
    last_checkpoint = find_last_checkpoint(Path(resume_path) / "checkpoints/train")  # TODO: Make this path configurable or global var
    if last_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {resume_path}")
    printer.info(f"Resuming from checkpoint: {last_checkpoint}")
    
    config = OmegaConf.load(resume_yaml)
    config.resume = resume_path
    config.resume_from_checkpoint = last_checkpoint

    return config


def from_engine_with_transform(keys: KeysCollection, transform: Transform, first: bool = False):
    """Utility function based on `monai.handlers.from_engine` that includes a transform.
    
    Transforms, specifically `monai.transforms.Compose`, will output a list of dicts over the
    batch, whereas many downstream handlers expect a tuple of lists,
    where each list contains all items for a batch, i.e. ([img1, img2, ...], [label1, label2, ...]).

    The alternative way, of using `monai.transforms.Compose` with from_engine directly,
    does not allow from_engine to unpack the outputs correctly,
    and would not be correctly typed anyhow.

    NOTE: Runs the transforms before extracting the keys, so the
    transforms should handle a dict.
    
    """

    from_engine_fn = from_engine(keys, first)

    def _wrapper(data):
        if isinstance(data, dict):
            # why does monai not use a class to ensure these things?
            data = [data]
        transformed = [transform(d) for d in data]
         # transformed is now a list of dicts, one per batch item
        return from_engine_fn(transformed)
    
    return _wrapper





def simple_math_resolver(expr):
    """Custom OmegaConf resolver to do simple math with + and -."""
    m = re.match(r"^\s*(-?\d+)\s*([\+\-])\s*(-?\d+)\s*$", expr)
    if not m:
        raise ValueError("Only single + or - supported: 'a + b' or 'a - b'")
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    return a + b if op == "+" else a - b


def state_dict_strip(state_dict: dict, target_prefix: str = "_orig_mod.") -> dict:
    """Strip given prefix from state dict keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(target_prefix):
            new_key = k[len(target_prefix) :]
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def state_dict_strip_compiled(state_dict: dict) -> dict:
    """Strip '_orig_mod.' prefix from state dict keys if present (from torch.compile)."""

    target_prefix = "_orig_mod."

    return state_dict_strip(state_dict, target_prefix)


def state_dict_strip_ddp(state_dict: dict) -> dict:
    """Strip 'module.' prefix from state dict keys if present (from DDP)."""

    target_prefix = "module."

    return state_dict_strip(state_dict, target_prefix)