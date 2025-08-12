from .base import PairedDataset

from .vocsegmentation import VOCSegmentationPairedDataset
from .monai import SegmentationFolder

# TODO: Move to hydra instantiate
def setup_dataset(config):
    match config.dataset.type:
        case "VOCSegmentationPairedDataset":
            return VOCSegmentationPairedDataset(
                config=config
            )
        case "MonaiSegmentationFolder":
            return SegmentationFolder(
                config=config
            )
        case _:
            raise ValueError(f"Dataset type {config.dataset.type} is not supported. It can be implemented in the datasets directory.")