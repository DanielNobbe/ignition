import os
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from ignition.data import setup_data
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from ignition.utils import save_config


@pytest.mark.skipif(os.getenv("RUN_SLOW_TESTS", 0) == 0, reason="Skip slow tests")
def test_setup_data():
    config = Namespace(data_path="~/data", batch_size=1, eval_batch_size=1, num_workers=0)
    dataloader_train, dataloader_eval = setup_data(config)

    assert isinstance(dataloader_train, DataLoader)
    assert isinstance(dataloader_eval, DataLoader)
    train_batch = next(iter(dataloader_train))
    assert isinstance(train_batch, dict)
    assert isinstance(train_batch["image"], Tensor)
    assert isinstance(train_batch["mask"], Tensor)
    assert train_batch["image"].ndim == 4
    assert train_batch["mask"].ndim == 3
    eval_batch = next(iter(dataloader_eval))
    assert isinstance(eval_batch, dict)
    assert isinstance(eval_batch["image"], Tensor)
    assert isinstance(eval_batch["mask"], Tensor)
    assert eval_batch["image"].ndim == 4
    assert eval_batch["mask"].ndim == 3


def test_save_config():
    with open("./config.yaml", "r") as f:
        config = OmegaConf.load(f)

    # Add backend to config (similar to setup_config)
    config.backend = None

    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)

        save_config(config, output_dir)

        with open(output_dir / "config-lock.yaml", "r") as f:
            test_config = OmegaConf.load(f)

        assert config == test_config
