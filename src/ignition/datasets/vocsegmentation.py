import torch

from PIL import Image
import numpy as np
from torchvision.datasets.voc import VOCSegmentation
import ignite.distributed as idist
from ignite.utils import convert_tensor

import albumentations as A
import cv2

from .abstract import IgnitionDataset, PairedDataset
from ignition.data import TransformedDataset, IgnoreMaskBoundaries, ToTensor

class VOCSegmentationPIL(VOCSegmentation):
    target_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    def __init__(self, *args, return_meta=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_meta = return_meta

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.images[index]).convert("RGB"))
        assert img is not None, f"Image at '{self.images[index]}' has a problem"
        mask = np.asarray(Image.open(self.masks[index]))

        if self.return_meta:
            return {
                "image": img,
                "mask": mask,
                "meta": {
                    "index": index,
                    "image_path": self.images[index],
                    "mask_path": self.masks[index],
                },
            }

        return {"image": img, "mask": mask}
    

class VOCSegmentationPairedDataset(PairedDataset):

    """This is implemented with the following steps:

    for train:
    1. load a VOCSegmentationPIL dataset with train split
    2. set up albumentations transforms
    3. create a TransformedDataset with the dataset and transforms
    4. create a DataLoader with the TransformedDataset
    
    """

    def _setup_datasets(self):
        dataset_train = VOCSegmentationPIL(
            root=self.config.dataset.root,
            year="2012",
            image_set="train",
            download=False,
        )
        
        dataset_eval = VOCSegmentationPIL(root=self.config.dataset.root, year="2012", image_set="val", download=False)

        val_img_size = self.config.dataset.val.img_size
        train_img_size = self.config.dataset.train.img_size

        mean = self.config.dataset.mean
        std = self.config.dataset.std

        transform_train = A.Compose(
            [
                A.RandomScale(scale_limit=(0.0, 1.5), interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(train_img_size, train_img_size),
                A.HorizontalFlip(),
                A.Blur(blur_limit=3),
                A.Normalize(mean=mean, std=std),
                IgnoreMaskBoundaries(),
                ToTensor(),
            ]
        )
        
        transform_eval = A.Compose(
            [
                A.PadIfNeeded(val_img_size, val_img_size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                IgnoreMaskBoundaries(),
                ToTensor(),
            ]
        )

        self.dataset_train = TransformedDataset(dataset_train, transform_fn=transform_train)
        self.dataset_eval = TransformedDataset(dataset_eval, transform_fn=transform_eval)

        if self.config.dataset.get('subset_size') is not None:
            train_subset_size = int(len(dataset_train) * self.config.dataset.subset_size)
            eval_subset_size = int(len(dataset_eval) * self.config.dataset.subset_size)
            if train_subset_size <= 0 or eval_subset_size <= 0:
                raise ValueError("data_subset_size must result in at least one sample for both subsets.")
            dataset_train = torch.utils.data.Subset(dataset_train, range(train_subset_size))
            dataset_eval = torch.utils.data.Subset(dataset_eval, range(eval_subset_size))


        self.train_dataloader = idist.auto_dataloader(
            self.dataset_train,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        self.val_dataloader = idist.auto_dataloader(
            self.dataset_eval,
            shuffle=False,
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

    def get_train_dataloader(self):
        return self.train_dataloader
    
    def get_eval_dataloader(self):
        return self.val_dataloader

    def get_prepare_batch(self):
        def prepare_batch(batch, device, non_blocking):
            x, y = batch["image"], batch["mask"]
            x = convert_tensor(x, device=device, non_blocking=non_blocking)
            y = convert_tensor(y, device=device, non_blocking=non_blocking).long()
            return x, y
        
        return prepare_batch