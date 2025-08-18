import sys
from functools import partial
from pprint import pformat
from typing import Any, cast
from warnings import warn

import hydra
import ignite.distributed as idist
import monai
from ignite.engine import Events
from ignite.handlers import LRScheduler, ProgressBar
from ignite.utils import manual_seed
from omegaconf import DictConfig

from ignition.data.utils import denormalize
from ignition.datasets import setup_dataset
from ignition.engines import setup_evaluator, setup_trainer
from ignition.handlers import setup_handlers
from ignition.losses import setup_loss
from ignition.lr_schedulers import setup_lr_scheduler
from ignition.metrics import setup_metrics
from ignition.models import setup_model
from ignition.optimizers import setup_optimizer
from ignition.utils import *

try:
    from torch.optim.lr_scheduler import LRScheduler as PyTorchLRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as PyTorchLRScheduler


def run(local_rank: int, config: Any):

    world_size = idist.get_world_size()
    if world_size > 1:
        warn(
            "Running this training script on multiple GPUs is not fully supported. To add support, look into adding rank-conditional logic for the handlers, metrics and dataloaders. And verify that all LR schedulers and optimizers support distributed training, e.g. by using `ignite.distributed.auto_optim`."
        )

    monai.config.print_config()
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder and copy config file to output dir
    output_dir = setup_output_dir(config, rank)
    if rank == 0:
        save_config(config, output_dir)

    config.output_dir = output_dir

    # load datasets and create dataloaders
    dataset = setup_dataset(config)
    dataloader_train = dataset.get_train_dataloader()
    dataloader_val = dataset.get_val_dataloader()
    le = len(dataloader_train)

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config))
    optimizer = idist.auto_optim(setup_optimizer(model.parameters(), config))
    loss_fn = setup_loss(config).to(device=device)
    lr_scheduler = setup_lr_scheduler(optimizer, config, le)

    # trainer and evaluator
    trainer = setup_trainer(
        config, model, optimizer, loss_fn, device, dataset, setup_metrics(config, "train", loss_fn, model)
    )
    validator = setup_evaluator(config, model, setup_metrics(config, "val", loss_fn, model), device, dataset)

    # setup engines logger with python logging
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(config))
    trainer.logger = validator.logger = logger

    setup_handlers(config, model, optimizer, trainer, validator, lr_scheduler=lr_scheduler)

    logger.info("Start training for %d epochs", config.max_epochs)

    # setup if done. let's run the training
    if config.engine_type == "ignite":
        trainer.run(
            dataloader_train,
            max_epochs=config.max_epochs,
            epoch_length=config.train_epoch_length,
        )
    elif config.engine_type == "monai":
        trainer.run()
    else:
        raise ValueError(f"Unknown engine type: {config.engine_type}. Supported types are 'ignite' and 'monai'.")


# main entrypoint
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    config = setup_config(cfg)
    with idist.Parallel(config.backend) as p:
        p.run(run, config=config)

    print("Training completed successfully!")


if __name__ == "__main__":
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra/job_logging=stdout")
    main()
