import os
from warnings import warn

from ignite.utils import convert_tensor
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityd,
    Spacingd,
    LabelFilterd
)

from .base import PairedDataset


class SegmentationFolder(PairedDataset):
    """
    A dataset class for loading images from a folder structure, where
    there are separate folders for images and labels.
    This class incorporates splitting the dataset into training and validation sets.
    """

    image_key = "image"
    label_key = "label"

    def _setup_datasets(self):
        # create a list of all files and label files in these directories
        # TODO: Implement another class that does not need to keep
        # items in memory
        # TODO: Implement based on CacheDataset

        self.images_dir = self.config.dataset.images_dir

        image_files = os.listdir(self.config.dataset.images_dir)

        if not self._verify_all_same_ext(image_files):
            raise ValueError(f"Not all image files in directory {self.config.images_dir} have the same extension.")

        self.labels_dir = self.config.dataset.labels_dir
        label_files = os.listdir(self.config.dataset.labels_dir)

        assert len(image_files) > 0, f"No image files found in directory {self.config.dataset.images_dir}."
        assert len(label_files) > 0, f"No label files found in directory {self.config.dataset.labels_dir}."

        if not self._verify_all_same_ext(label_files):
            raise ValueError(f"Not all label files in directory {self.config.labels_dir} have the same extension.")

        data_dict = [self._create_dict(image_file) for image_file in image_files]

        # super().__init__(data=data_dict)

        # TODO: Handle subsetting to decrease size

        # split into train and eval sets
        if self.config.dataset.get("val_size") is not None:
            val_size = self.config.dataset.val_size
        else:
            val_size = 0.2
            warn("val_size not specified in config, defaulting to 0.2 (20% of data will be used for evaluation).")
        if not (0 < val_size < 1):
            raise ValueError("val_size must be a float between 0 and 1.")
        split_index = int(len(data_dict) * (1 - val_size))
        self.train_data = data_dict[:split_index]
        self.val_data = data_dict[split_index:]

        # TODO: Use cached or threaddataset from monai
        # train_images, train_labels = zip(*[(item['img'], item.get('label')) for item in self.train_data])
        # val_images, val_labels = zip(*[(item['img'], item.get('label')) for item in self.val_data])
        if self.config.dataset.get("subset_size") is not None and (self.config.dataset.subset_size < 1.0):
            warn("Using subset_size, which does not use torch.Subset, but takes a naive slice.")
            train_subset_size = int(len(self.train_data) * self.config.dataset.subset_size)
            eval_subset_size = int(len(self.val_data) * self.config.dataset.subset_size)
            if train_subset_size <= 0 or eval_subset_size <= 0:
                raise ValueError("data_subset_size must result in at least one sample for both subsets.")
            self.train_data = self.train_data[:train_subset_size]
            self.val_data = self.val_data[:eval_subset_size]

        self.train_dataset = CacheDataset(
            data=self.train_data,
            transform=self._get_train_transforms(),
            cache_num=self.config.dataset.get("cache_num", 1),  # number of samples to cache in memory
        )
        self.val_dataset = CacheDataset(
            data=self.val_data,
            transform=self._get_val_transforms(),
            cache_num=self.config.dataset.get("cache_num", 1),  # number of samples to cache in memory
        )

        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples.")

        # TODO: Add train and val transforms

        # NOTE: MONAI training systems often use sliding window inference
        # which works well to get models that can infer on images with irregular
        # size, especially if the number of layers varies.
        # it does cost more compute to train, although
        # probably the performance is bad if evaluating with sliding
        # window while training on crops?
        # TODO: Add option to use sliding window inference, compare models

        # NOTE: By using MONAI Dataloader,
        # we are incompatible with multi-node training,
        # but multi-GPU training should work fine.
        self.train_dataloader = DataLoader(
            self.train_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if self.config.get("inferer") is not None:
            # if using custom inferer, we use batch size of 1,
            # since it may do custom batching
            eval_batch_size = 1
            warn(
                f"Using custom inferer {self.config.inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
        else:
            # otherwise, we use the eval batch size from the config
            eval_batch_size = self.config.eval_batch_size

        self.val_dataloader = DataLoader(
            self.val_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=eval_batch_size,  # if using sliding window inference, batch size is 1
            shuffle=False,
            drop_last=False,
        )

    def _get_transform(self, transform: dict):
        # TODO: Move to transforms file
        # TODO: Move to hydra instantiate
        # Selected transforms are based on https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb
        match transform.type:
            case "LabelFilter":
                return LabelFilterd(
                    keys=[self.label_key],
                    applied_labels=transform.get("include_classes", 0),
                )
            case "AsDiscrete":
                return AsDiscreted(
                    keys=[self.label_key],
                    argmax=transform.get("argmax", True),
                    to_onehot=transform.get("to_onehot", None),
                    include_background=transform.get("include_background", True),
                )
            case "RandSpatialCrop":
                return RandSpatialCropd(
                    keys=[self.image_key, self.label_key],
                    roi_size=transform.roi_size,
                    random_center=transform.get("random_center", True),
                    random_size=transform.get("random_size", False),
                )
            case "ScaleIntensity":
                return ScaleIntensityd(
                    keys=[self.image_key],
                    minv=transform.get("minv", 0.0),
                    maxv=transform.get("maxv", 1.0),
                    factor=transform.get("factor", None),
                    channel_wise=transform.get("channel_wise", False),
                )
            case "Orientation":
                return Orientationd(
                    keys=[self.image_key, self.label_key], axcodes=transform.axcodes
                )  # TODO: CHeck how we use this
            case "Spacing":
                return Spacingd(
                    keys=[self.image_key, self.label_key],
                    pixdim=transform.pixdim,
                    mode=transform.get("mode", "bilinear"),
                    align_corners=transform.get("align_corners", None),
                )
            case "RandFlip":
                return RandFlipd(
                    keys=[self.image_key, self.label_key],
                    prob=transform.prob,
                    spatial_axis=transform.get("spatial_axis", None),
                )
            case "NormalizeIntensity":
                return NormalizeIntensityd(
                    keys=[self.image_key],
                    nonzero=transform.get("nonzero", False),
                    channel_wise=transform.get("channel_wise", False),
                    subtrahend=transform.get("subtrahend", None),
                    divisor=transform.get("divisor", None),
                )
            case "RandScaleIntensity":
                return RandScaleIntensityd(
                    keys=[self.image_key],
                    factors=transform.factors,
                    prob=transform.prob,
                    channel_wise=transform.get("channel_wise", False),
                )
            case "RandShiftIntensity":
                return RandShiftIntensityd(
                    keys=[self.image_key],
                    offsets=transform.offsets,
                    prob=transform.prob,
                    channel_wise=transform.get("channel_wise", False),
                )
            case "EnsureChannelFirst":
                return EnsureChannelFirstd(keys=[self.image_key, self.label_key], channel_dim=transform.channel_dim)
            case _:
                raise ValueError(f"Unknown transform type: {transform.type}")

    def _get_train_transforms(self):
        transforms = []

        transforms.append(LoadImaged(keys=[self.image_key, self.label_key]))
        transforms.append(EnsureTyped(keys=[self.image_key, self.label_key]))

        # TODO: In BraTS, they use channels per label class, do we need this too?

        for transform in self.config.dataset.transforms.get("train", []):
            transforms.append(self._get_transform(transform))

        return Compose(transforms)

    def _get_val_transforms(self):
        transforms = []

        transforms.append(LoadImaged(keys=[self.image_key, self.label_key]))
        transforms.append(EnsureTyped(keys=[self.image_key, self.label_key]))

        for transform in self.config.dataset.transforms.get("val", []):
            transforms.append(self._get_transform(transform))

        return Compose(transforms)

    @staticmethod
    def _verify_all_same_ext(files: list[str]) -> bool:
        """Check if all files have the same extension."""
        if not files:
            return True
        ext = os.path.splitext(files[0])[1]
        return all(os.path.splitext(f)[1] == ext for f in files)

    def _create_dict(self, image_file):
        output = {self.image_key: os.path.join(self.images_dir, image_file)}

        if self.labels_dir:
            file_name = image_file  # could also split off extension if it's different
            output[self.label_key] = os.path.join(self.labels_dir, file_name)

            # ensure it exists
            if not os.path.isfile(output[self.label_key]):
                raise ValueError(f"Label file {output[self.label_key]} not found.")

        return output

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_val_dataloader(self):
        return self.val_dataloader

    def get_prepare_batch(self):
        def prepare_batch(batch, device, non_blocking):
            x, y = batch[self.image_key], batch[self.label_key]
            return x, y

        return prepare_batch
