from .base import PairedDataset, IgnitionDataset
from .monai import SegmentationFolder, EvalSegmentationFolder
from .vocsegmentation import VOCSegmentationPairedDataset
from .ri import RiDatasetFromFile, RiSingleDatasetFromFile


# TODO: Move to hydra instantiate
def setup_dataset(config, get_test: bool = False):
    
    match config.dataset.type:
        case "VOCSegmentationPairedDataset":
            return VOCSegmentationPairedDataset(config=config)
        case "MonaiSegmentationFolder":
            if get_test:
                raise NotImplementedError()
            return SegmentationFolder(config=config)
        case "MonaiEvalSegmentationFolder":
            return EvalSegmentationFolder(config=config)
        case "RiDatasetFromFile":
            if get_test:
                return RiSingleDatasetFromFile(config=config)
            return RiDatasetFromFile(config=config)
        case "RiSingleDatasetFromFile":
            return RiSingleDatasetFromFile(config=config)
        case _:
            raise ValueError(
                f"Dataset type {config.dataset.type} is not supported. It can be implemented in the datasets directory."
            )
