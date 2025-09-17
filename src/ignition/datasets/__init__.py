from .base import PairedDataset, IgnitionDataset
from .monai import SegmentationFolder, EvalSegmentationFolder
from .vocsegmentation import VOCSegmentationPairedDataset


# TODO: Move to hydra instantiate
def setup_dataset(config):
    match config.dataset.type:
        case "VOCSegmentationPairedDataset":
            return VOCSegmentationPairedDataset(config=config)
        case "MonaiSegmentationFolder":
            return SegmentationFolder(config=config)
        case "MonaiEvalSegmentationFolder":
            return EvalSegmentationFolder(config=config)
        case _:
            raise ValueError(
                f"Dataset type {config.dataset.type} is not supported. It can be implemented in the datasets directory."
            )
