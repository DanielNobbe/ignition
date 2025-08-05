from .vocsegmentation import VOCSegmentationPairedDataset

def setup_dataset(config):
    if config.dataset.type == "VOCSegmentationPairedDataset":
        return VOCSegmentationPairedDataset(
            config=config,
        )
    else:
        raise ValueError(f"Dataset type {config.dataset.type} is not supported. It can be implemented in the datasets directory.")