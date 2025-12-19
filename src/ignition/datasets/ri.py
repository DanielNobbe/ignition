import os
from pathlib import Path
from warnings import warn

from hydra.utils import instantiate
from omegaconf import DictConfig
import ignite.distributed as idist
from ignite.utils import convert_tensor
from monai.data import CacheDataset, LMDBDataset, DataLoader
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
    LabelFilterd,
    MapLabelValued,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandAdjustContrastd,
    AsDiscreted,
    SpatialPadd,
    CenterSpatialCropd
)

from .base import PairedDataset, IgnitionDataset
from .mixins import MonaiTransformsMixin, MonaiDatasetUtilsMixin
from ignition.transforms import RiCheckLabelMap, MapKeysTransform, MakeAbsPathd

from logging import getLogger

import json

import numpy as np

printer = getLogger(__name__)


class RiDatasetMixin():
    image_key: str = "image"
    label_key: str = "label"

    @staticmethod
    def _load_dataset_file(dataset_file: str):
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        return data
    
    def get_prepare_batch(self):
        def prepare_batch(batch, device=None, non_blocking=None):
            x, y = batch[self.image_key], batch.get(self.label_key)
            return x, y

        return prepare_batch
    
    def _make_abs_path(self, item: str | list | dict, optional: bool = False) -> str | list | dict:
        if isinstance(item, (str, os.PathLike)):
            hypothetical_path = os.path.join(self.data_root, item)
            if optional and not os.path.isfile(hypothetical_path):
                return item
            return os.path.join(self.data_root, item)
        elif isinstance(item, list):
            return [self._make_abs_path(f) for f in item]
        elif isinstance(item, dict):
            return {k: self._make_abs_path(v, optional=True) for k, v in item.items()}
        else:
            if not optional:
                raise ValueError(f"Unsupported type for path in dataset item: {type(item)}")
            return item
        
    def _make_abs_paths(self, item: dict, keys: list[str]) -> dict:
        """Convert relative paths to absolute paths based on self.data_root."""
        # NOTE: Could make this a transform too
        for key in keys:
            if key in item:
                item[key] = self._make_abs_path(item[key])
            # if isinstance(item[file_type], str | os.PathLike):
            #     item[key] = os.path.join(self.data_root, item[key])
            # elif isinstance(item[key], list):
            #     item[key] = [self._make_abs_path() for f in item[key]]
            # else:
            #     raise ValueError(f"Unsupported type for {file_type} in dataset item: {type(item[file_type])}")
        return item
    
    @staticmethod
    def _check_for_splits_as_datalist(data_file: str) -> bool:
        data_dict = __class__._load_dataset_file(data_file)
        if isinstance(data_dict, dict | DictConfig):
            # check for train and val keys
            if "training" in data_dict.keys() and "validation" in data_dict.keys():
                warn("dataset_file seems to be in datalist format with 'training' and 'validation' keys.")
                return True
        return False
    
    def _print_split_warning(self):
        if self.config.dataset.get("val_size") is not None:
            warn("val_size specified in config, but config defines split (either through dataset file or through separate files).")
            
    def _set_data_keys_from_cfg(self):
        self.data_image_key = self.config.dataset.get("image_key", "image")
        self.data_label_key = self.config.dataset.get("label_key", "label")
        self.patient_id_key = self.config.dataset.get("patient_id_key", "patient_id")
        self.label_map_key = self.config.dataset.get("label_map_key", "label_map")

    def _get_data_keys(self) -> list[str]:
        return [self.data_image_key, self.data_label_key, self.patient_id_key, self.label_map_key]
        

class RiDatasetFromFile(RiDatasetMixin, PairedDataset, MonaiTransformsMixin, MonaiDatasetUtilsMixin):
    """
    A dataset class for loading images and labels from a JSON file defininf the dataset.
    The main argument that needs to be specified in the config is:
    dataset_file: path to the JSON file defining the dataset.
    The Ri dataset definition file has the following structure:

    {
        <global_id>: {
            "id": <global_id>,
            "image": "<path_to_image_file>", # can be other key, such as 'mr'
            "label": "<path_to_label_file>" # can be other key, such as 'segmentation'
            "label_map": <int-to-name mapping for labels>  # should verify that they are all the same
            "patient_id": <global_patient_id>,  # need to ensure patients are not split across train/val
            "sequence_type": <sequence_type>  # optional, unused for now
            "original_dataset": <original_dataset_name>  # optional, unused for now
            "modality": <modality_type>  # optional, unused for now
            }    
    }
    Alternatively, it can also be a single list of all items, without the global_id keys.
    We also support datalist-style files with 'training' and 'validation' keys.

    We support the following dataset file configurations:
    1. A dict specified in the config, with 'train' and 'val' keys pointing to separate JSON files.
    2. A single JSON file with 'training' and 'validation' keys.
    3. A single JSON file with all data, which will be split into train and val sets based on val_size.

    TODO: Implement a label mapping transform that uses the label_map on-the-fly.

    -> initially, we will just check that all label maps are the same during data loading.

    """

    def _sample_val_split(self, all_keys: list[str], val_size: float) -> tuple[list[str], list[str]]:
        rng = np.random.default_rng(self.config.get("seed", 42))
        number_of_val_samples = int(len(all_keys) * val_size)

        sorted_all_keys = sorted(all_keys)  # ensures deterministic behaviour across runs
        val_keys = rng.choice(sorted_all_keys, size=number_of_val_samples, replace=False).tolist()
        train_keys = [key for key in all_keys if key not in val_keys]
        return train_keys, val_keys

    def _setup_datasets(self):
        self._set_data_keys_from_cfg()
        # print(f"Using image key: {self.data_image_key}, label key: {self.data_label_key}, patient id key: {self.patient_id_key}, label map key: {self.label_map_key}")

        if isinstance(self.config.dataset.dataset_file, dict | DictConfig):
            # we allow specifying two JSON files, one for train and one for val
            assert 'train' in self.config.dataset.dataset_file.keys() and 'val' in self.config.dataset.dataset_file.keys(), "If dataset_file is a dict, it must contain 'train' and 'val' keys."
            train_file = self.config.dataset.dataset_file.train
            val_file = self.config.dataset.dataset_file.val

            self.train_data = self._load_dataset_file(train_file)
            self.val_data = self._load_dataset_file(val_file)

            if isinstance(self.train_data, dict):
                self.train_data = list(self.train_data.values())
            if isinstance(self.val_data, dict):
                self.val_data = list(self.val_data.values())

            self._print_split_warning()
        elif self._check_for_splits_as_datalist(self.config.dataset.dataset_file):
            data_dict = self._load_dataset_file(self.config.dataset.dataset_file)
            assert "training" in data_dict.keys() and "validation" in data_dict.keys(), "dataset_file must contain 'training' and 'validation' keys."
            self.train_data = data_dict["training"]
            self.val_data = data_dict["validation"]
            if isinstance(self.train_data, dict):
                self.train_data = list(self.train_data.values())
            if isinstance(self.val_data, dict):
                self.val_data = list(self.val_data.values())
            
            self._print_split_warning()
        else:
            self.data_dict = self._load_dataset_file(self.config.dataset.dataset_file)

            # since we have heterogeneous dataset, and need to maintain a deterministic split,
            # we cannot sort the keys, since they include the dataset name.
            # instead, we should uniformly sample from the keys.

            # split into train and eval sets
            if self.config.dataset.get("val_size") is not None:
                val_size = self.config.dataset.val_size
            else:
                val_size = 0.2
                warn("val_size not specified in config, defaulting to 0.2 (20% of data will be used for evaluation).")
            if not (0 < val_size < 1):
                raise ValueError("val_size must be a float between 0 and 1.")
            
            if self.config.dataset.get("split_by_patient", True):
                warn("splitting by patient, not by sample ID for validation set.")
                all_patient_ids = {item[self.patient_id_key] for item in self.data_dict.values()}
                train_patient_ids, val_patient_ids = self._sample_val_split(list(all_patient_ids), val_size)
                warn(f"Using {len(train_patient_ids)} train patients and {len(val_patient_ids)} val patients. Splitting by exact number of samples is not guaranteed.")
                train_keys = [key for key, item in self.data_dict.items() if item[self.patient_id_key] in train_patient_ids]
                val_keys = [key for key, item in self.data_dict.items() if item[self.patient_id_key] in val_patient_ids]
                # print(f"Train samples: {len(train_keys)}, Val samples: {len(val_keys)} ({len(val_keys)/(len(train_keys)+len(val_keys)):.2%} of total samples for validation)")
            else:
                all_keys = list(self.data_dict.keys())
                train_keys, val_keys = self._sample_val_split(all_keys, self.config.dataset.val_size)
            
            self.train_data = [self.data_dict[key] for key in train_keys]
            self.val_data = [self.data_dict[key] for key in val_keys]

        self.include_metadata_keys = self.config.dataset.get("include_metadata_keys", [])
        self.data_root = self.config.dataset.get("data_root", None)
        if self.data_root is None:
            self.data_root = os.path.dirname(self.config.dataset.dataset_file)
            warn(f"data_root not specified in config, defaulting to dataset file directory: {self.data_root}")

        self.train_data = [self._make_abs_paths(item, keys=[self.data_image_key, self.data_label_key]) for item in self.train_data]
        self.val_data = [self._make_abs_paths(item, keys=[self.data_image_key, self.data_label_key]) for item in self.val_data]

        # filter out some metadata since some keys can be incompatible with collating
        self.train_data = self._filter_data_by_keys(self.train_data, self.include_metadata_keys + self._get_data_keys())
        self.val_data = self._filter_data_by_keys(self.val_data, self.include_metadata_keys + self._get_data_keys())

        if self.config.dataset.get("check_label_map", True):
            self.first_label_map = self.train_data[0].get(self.label_map_key)
            if self.first_label_map is None:
                print(self.train_data[0].keys())
                raise ValueError("No label_map found in the first training sample.")

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

        if self.config.dataset.get('save_dataset_splits'):
            # just save it all
            train_save_path = Path(self.config.output_dir) / f"train_dataset.json"
            with open(train_save_path, 'w') as f:
                json.dump(self.train_data, f, indent=4)
            printer.info(f"Saved train dataset to {train_save_path}")
            val_save_path = Path(self.config.output_dir) / f"val_dataset.json"
            with open(val_save_path, 'w') as f:
                json.dump(self.val_data, f, indent=4)
            printer.info(f"Saved val dataset to {val_save_path}")

        self.train_dataset = self.build_dataset(self.config.dataset, self.train_data, self._get_train_transforms())

        self.val_dataset = self.build_dataset(self.config.dataset, self.val_data, self._get_val_transforms())

        printer.info(f"Loaded {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples.")


        if self.config.get("inferer") is not None:
            # if using custom inferer, we use batch size of 1,
            # since it may do custom batching
            self.config.eval_batch_size = 1
            warn(
                f"Using custom inferer {self.config.inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )

        if self.config.get("train_inferer", False):
            warn(
                f"Using custom inferer {self.config.train_inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
            self.config.batch_size = 1


    def _get_train_transforms(self):
        transforms = []

        if self.config.dataset.get("check_label_map", True):
            transforms.append(
                RiCheckLabelMap(
                    expected_label_map=self.first_label_map,
                    label_map_key=self.label_map_key
                )
            )
        
        transforms.append(
            MapKeysTransform(
                key_mapping={
                    self.data_image_key: self.image_key,
                    self.data_label_key: self.label_key
                },
                remove_old_keys=True
            )
        )
        if self.config.dataset.get("auto_add_loadimage", True):
            transforms.append(LoadImaged(keys=[self.image_key, self.label_key]))
            transforms.append(EnsureTyped(keys=[self.image_key, self.label_key]))

        for transform in self.config.dataset.transforms.get("train", []):
            transforms.append(self._get_transform(transform))

        return Compose(transforms)

    def _get_val_transforms(self):
        transforms = []

        if self.config.dataset.get("check_label_map", True):
            transforms.append(
                RiCheckLabelMap(
                    expected_label_map=self.first_label_map,
                    label_map_key=self.label_map_key
                )
            )

        transforms.append(
            MapKeysTransform(
                key_mapping={
                    self.data_image_key: self.image_key,
                    self.data_label_key: self.label_key
                },
                remove_old_keys=True
            )
        )

        if self.config.dataset.get("auto_add_loadimage", True):
            transforms.append(LoadImaged(keys=[self.image_key, self.label_key]))
            transforms.append(EnsureTyped(keys=[self.image_key, self.label_key]))

        for transform in self.config.dataset.transforms.get("val", []):
            transforms.append(self._get_transform(transform))

        return Compose(transforms)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.config.batch_size,  # if using sliding window inference, batch size is 1
            shuffle=True,
            drop_last=True,
        )

    def get_val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.config.eval_batch_size,  # if using sliding window inference, batch size is 1
            shuffle=False,
            drop_last=False,
        )
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_val_dataset(self):
        return self.val_dataset

        
class RiSingleDatasetFromFile(
        IgnitionDataset,
        MonaiTransformsMixin,
        MonaiDatasetUtilsMixin,
        RiDatasetMixin
    ):
    """A single dataset class for loading from a JSON file. For e.g. test dataset.
    
    We support the following dataset file configurations:
    1. A single JSON file with all data, which will be used as is.
    2. A single JSON file with 'training' and 'validation' keys.
    
    We do not support automatic splitting here, to discourage users from randomly sampling test sets,
    as this should be pre-defined.
    
    """

    
    def _get_transforms(self):
        transforms = []

        if self.config.dataset.get("check_label_map", True):
            transforms.append(
                RiCheckLabelMap(
                    expected_label_map=self.first_label_map,
                    label_map_key=self.label_map_key
                )
            )

        transforms.append(
            MapKeysTransform(
                key_mapping={
                    self.data_image_key: self.image_key,
                    self.data_label_key: self.label_key
                },
                remove_old_keys=True
            )
        )

        if self.config.dataset.get("auto_add_loadimage", True):
            transforms.append(LoadImaged(keys=[self.image_key, self.label_key]))
            transforms.append(EnsureTyped(keys=[self.image_key, self.label_key]))
            # transforms.append(MakeAbsPathd(keys=[self.image_key, self.label_key], root_dir=self.data_root))

        for transform in self.config.dataset.transforms:
            transforms.append(self._get_transform(transform))

        return Compose(transforms)

    def _setup_dataset(self):
        self._set_data_keys_from_cfg()

        if self._check_for_splits_as_datalist(self.config.dataset.dataset_file) or self.config.dataset.get("datalist_key") is not None:
            datalist_key = self.config.dataset.get("datalist_key", "validation")
            data_dict = self._load_dataset_file(self.config.dataset.dataset_file)
            assert datalist_key in data_dict.keys(), f"dataset_file must contain '{datalist_key}' key. If using a different key, please specify the `datalist_key` in the config."
            self.data_dict = data_dict[datalist_key]
            if isinstance(self.data_dict, dict):
                self.data_dict = list(self.data_dict.values())
            self._print_split_warning()
        else:
            # we use the entire JSON file as is
            self.data_dict = self._load_dataset_file(self.config.dataset.dataset_file)
            if isinstance(self.data_dict, dict):
                self.data_dict = list(self.data_dict.values())
        self.include_metadata_keys = self.config.dataset.get("include_metadata_keys", [])

        self.data_root = self.config.dataset.get("data_root", None)
        if self.data_root is None:
            self.data_root = os.path.dirname(self.config.dataset.dataset_file)
            warn(f"data_root not specified in config, defaulting to dataset file directory: {self.data_root}")

        # if isinstance(self.data_dict, list):
        self.data = [self._make_abs_paths(item, keys=[self.data_image_key, self.data_label_key]) for item in self.data_dict]
        # elif isinstance(self.data_dict, dict):
        #     self.data = [self._make_abs_paths(item) for item in self.data_dict.values()]
        # cannot be anything else in JSON?

        self.data = self._filter_data_by_keys(self.data, self.include_metadata_keys + self._get_data_keys())
        if self.config.dataset.get("check_label_map", True):
            self.first_label_map = self.data[0].get(self.label_map_key, None)
            if self.first_label_map is None:
                raise ValueError(f"No label map (key: {self.label_map_key}) found in the first sample.")

        self.dataset = self.build_dataset(self.config.dataset, self.data, self._get_transforms())

        printer.info(f"Loaded {len(self.dataset)} samples.")

        if self.config.get("inferer") is not None:
            # if using custom inferer, we use batch size of 1,
            # since it may do custom batching
            self.config.eval_batch_size = 1
            warn(
                f"Using custom inferer {self.config.inferer.get('_target_', 'unknown')}, setting eval batch size to 1. Inferer may handle batching. "
            )
    
    def get_dataset(self):
        return self.dataset
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            num_workers=self.config.get("num_workers", 1),
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )