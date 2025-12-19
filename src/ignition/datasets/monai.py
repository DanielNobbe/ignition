import os
from warnings import warn

import ignite.distributed as idist
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged
)

from .base import PairedDataset, IgnitionDataset
from .mixins import MonaiFolderUtilsMixin, MonaiTransformsMixin, MonaiDatasetUtilsMixin

from logging import getLogger

printer = getLogger(__name__)


class SegmentationFolder(MonaiFolderUtilsMixin, PairedDataset, MonaiTransformsMixin, MonaiDatasetUtilsMixin):
    """
    A dataset class for loading images from a folder structure, where
    there are separate folders for images and labels.
    This class incorporates splitting the dataset into training and validation sets.
    """

    def _setup_datasets(self):
        # create a list of all files and label files in these directories
        # TODO: Implement another class that does not need to keep
        # items in memory
        # TODO: Implement based on CacheDataset

        self.images_dir = self.config.dataset.images_dir

        image_files = self._filter_image_files(os.listdir(self.config.dataset.images_dir))

        image_files.sort()  # ensure consistent order for validation split

        printer.info(f"All image files: {image_files}")

        if not self._verify_all_same_ext(image_files):
            raise ValueError(f"Not all image files in directory {self.config.dataset.images_dir} have the same extension.")

        self.labels_dir = self.config.dataset.labels_dir
        label_files = self._filter_image_files(os.listdir(self.config.dataset.labels_dir))

        assert len(image_files) > 0, f"No image files found in directory {self.config.dataset.images_dir}."
        assert len(label_files) > 0, f"No label files found in directory {self.config.dataset.labels_dir}."


        if not self._verify_all_same_ext(label_files):
            raise ValueError(f"Not all label files in directory {self.config.labels_dir} have the same extension.")

        data_list = [self._create_dict(image_file) for image_file in image_files]

        # super().__init__(data=data_list)  # TODO: Move logic to parent class

        # TODO: Handle subsetting to decrease size

        # split into train and eval sets
        if self.config.dataset.get("val_size") is not None:
            val_size = self.config.dataset.val_size
        else:
            val_size = 0.2
            warn("val_size not specified in config, defaulting to 0.2 (20% of data will be used for evaluation).")
        if not (0 < val_size < 1):
            raise ValueError("val_size must be a float between 0 and 1.")
        split_index = int(len(data_list) * (1 - val_size))
        self.train_data = data_list[:split_index]
        self.val_data = data_list[split_index:]

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
        
        if idist.get_world_size() > 1:
            if not self.list_match_across_ranks(self.train_data, tag="train_data") or not self.list_match_across_ranks(self.val_data, tag="val_data"):
                raise ValueError("Train and validation data lists do not match across ranks.")

        # self.train_dataset = CacheDataset(
        #     data=self.train_data,
        #     transform=self._get_train_transforms(),
        #     cache_num=self.config.dataset.get("cache_num", 1),  # number of samples to cache in memory
        # )
        self.train_dataset = self.build_dataset(self.config.dataset, self.train_data, self._get_train_transforms())
        # self.val_dataset = CacheDataset(
        #     data=self.val_data,
        #     transform=self._get_val_transforms(),
        #     cache_num=self.config.dataset.get("cache_num", 1),  # number of samples to cache in memory
        # )
        self.val_dataset = self.build_dataset(self.config.dataset, self.val_data, self._get_val_transforms())

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

        if self.config.get("inferer") is not None:
            # if using custom inferer, we use batch size of 1,
            # since it may do custom batching
            self.eval_batch_size = 1
            warn(
                f"Using custom inferer {self.config.inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
        else:
            # otherwise, we use the eval batch size from the config
            self.eval_batch_size = self.config.eval_batch_size

        if self.config.get("train_inferer", False):
            warn(
                f"Using custom inferer {self.config.train_inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
            self.train_batch_size = 1
        else:
            self.train_batch_size = self.config.batch_size


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

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.train_batch_size,  # if using sliding window inference, batch size is 1
            shuffle=True,
            drop_last=True,
        )

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.eval_batch_size,  # if using sliding window inference, batch size is 1
            shuffle=False,
            drop_last=False,
        )
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset


class EvalSegmentationFolder(IgnitionDataset, MonaiTransformsMixin, MonaiFolderUtilsMixin, MonaiDatasetUtilsMixin):
    """
    A dataset class for evaluation, which does include labels, but only contains a single dataset.
    """

    label_key = "label"  # still need to define, but will not be used

    def _setup_dataset(self):
        # create a list of all files and label files in these directories
        self.images_dir = self.config.dataset.images_dir

        image_files = self._filter_image_files(os.listdir(self.config.dataset.images_dir))

        image_files.sort()  # ensure consistent order for validation split

        printer.info(f"All image files: {image_files}")

        if not self._verify_all_same_ext(image_files):
            raise ValueError(f"Not all image files in directory {self.config.dataset.images_dir} have the same extension.")

        self.labels_dir = self.config.dataset.get('labels_dir')
        if self.labels_dir is not None:
            label_files = self._filter_image_files(os.listdir(self.config.dataset.labels_dir))
            assert len(label_files) > 0, f"No label files found in directory {self.config.dataset.labels_dir}."
            if not self._verify_all_same_ext(label_files):
                raise ValueError(f"Not all label files in directory {self.config.labels_dir} have the same extension.")

        assert len(image_files) > 0, f"No image files found in directory {self.config.dataset.images_dir}."



        data_list = [self._create_dict(image_file, self.labels_dir) for image_file in image_files]

        if self.config.dataset.get("subset_size") is not None and (self.config.dataset.subset_size < 1.0):
            warn("Using subset_size, which does not use torch.Subset, but takes a naive slice.")
            subset_size = int(len(data_list) * self.config.dataset.subset_size)

            if subset_size <= 0:
                raise ValueError("data_subset_size must result in at least one sample.")
            data_list = data_list[:subset_size]

        # TODO: Move further logic to a parent class

        # self.dataset = CacheDataset(
        #     data=data_list,
        #     transform=self._get_transforms(),
        #     cache_num=self.config.dataset.get("cache_num", 1),  # number of samples to cache in memory
        # )

        self.dataset = self.build_dataset(self.config.dataset, data_list, self._get_transforms())

        printer.info(f"Loaded {len(self.dataset)} samples.")

        if self.config.get("inferer") is not None:
            # if using custom inferer, we use batch size of 1,
            # since it may do custom batching
            self.config.eval_batch_size = 1
            warn(
                f"Using custom inferer {self.config.inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
        else:
            # otherwise, we use the eval batch size from the config
            self.config.eval_batch_size = self.config.eval_batch_size

    def _get_transforms(self):
        transforms = []

        transforms.append(LoadImaged(keys=[self.image_key, self.label_key], allow_missing_keys=True))
        transforms.append(EnsureTyped(keys=[self.image_key, self.label_key], allow_missing_keys=True))

        for transform in self.config.dataset.get("transforms", []):
            transforms.append(self._get_transform(transform))

        return Compose(transforms)
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
    
    def get_dataset(self):
        return self.dataset
        
    
