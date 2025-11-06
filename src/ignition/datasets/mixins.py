import os
from pathlib import Path
from warnings import warn

from hydra.utils import instantiate
from ignite.utils import convert_tensor

from monai.data import Dataset, CacheDataset, LMDBDataset, DataLoader
from monai.data.utils import pickle_hashing
from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
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
    # RandZoomd,
)

class MonaiTransformsMixin:
    # TODO: Figure out how to ensure the self.image_key and self.label_key
    # are present in the child class
    def _get_transform(self, transform: dict):
        # TODO: Move to transforms file
        # TODO: Move to hydra instantiate
        # Selected transforms are based on https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/brats_segmentation_3d.ipynb

        if '_target_' in transform.keys():
            hy_config = transform.copy()
            return instantiate(hy_config)

        match transform.type:
            case "SpatialPad":
                return SpatialPadd(
                    keys=[self.image_key, self.label_key],
                    spatial_size=transform.spatial_size,  # should be at least the roi size
                    mode=transform.get("mode", "constant"),  # padding mode
                    constant_values=transform.get("constant_values", 0),  # padding value
                )
            case "MapLabelValue":
                return MapLabelValued(
                    keys=[self.label_key],
                    orig_labels=transform.orig_labels,
                    target_labels=transform.target_labels,
                )
            case "LabelFilter":
                return LabelFilterd(
                    keys=[self.label_key],
                    applied_labels=transform.get("applied_labels", 0),
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
            case "RandGaussianSmooth":
                return RandGaussianSmoothd(
                    keys=[self.image_key],
                    prob=transform.prob,
                    sigma_x=transform.get("sigma_x", (0.5, 1.5)),
                    sigma_y=transform.get("sigma_y", (0.5, 1.5)),
                    sigma_z=transform.get("sigma_z", (0.5, 1.5)),
                )
            case "RandGaussianNoise":
                return RandGaussianNoised(
                    keys=[self.image_key],
                    prob=transform.prob,
                    mean=transform.get("mean", 0.0),
                    std=transform.get("std", 0.5),
                )
            case "RandAdjustContrast":
                return RandAdjustContrastd(
                    keys=[self.image_key],
                    prob=transform.prob,
                    gamma=transform.get("gamma", (0.7, 1.5)),
                    retain_stats=transform.get("retain_stats", True),
                    invert_image=transform.get("invert_image", False),
                )
            case "AsDiscreteLabel":
                return AsDiscreted(
                    keys=[self.label_key],
                    argmax=transform.get("argmax", False),
                    to_onehot=transform.get("to_onehot", None),
                    include_background=transform.get("include_background", True),
                )
            case _:
                raise ValueError(f"Unknown transform type: {transform.type}")


class MonaiFolderUtilsMixin:
    image_key = "image"
    label_key = "label"

    image_extensions = [
        ".nii",
        ".nii.gz",
    ]

    @staticmethod
    def _get_full_extension(file: str) -> str:
        return ''.join(Path(file).suffixes)

    def _filter_image_files(self, files: list[str]) -> list[str]:
        """Filter image files based on the defined extensions."""
        return [f for f in files if self._get_full_extension(f) in self.image_extensions]
    
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
    
    def get_prepare_batch(self):
        def prepare_batch(batch, device, non_blocking):
            x, y = batch[self.image_key], batch[self.label_key]
            return x, y

        return prepare_batch
        
class MonaiDatasetUtilsMixin:
    @staticmethod
    def build_dataset(dataset_config, data_list, transforms):
        match dataset_config.get("cache_mode", "memory"):
            case "memory":
                if not dataset_config.get("cache_num", False):
                    raise ValueError("cache_num must be specified in config when using memory cache_mode.")
                return CacheDataset(
                    data=data_list,
                    transform=transforms,
                    cache_num=dataset_config.get("cache_num", 1),  # number of samples to cache in memory
                )
            case "lmdb":
                if not dataset_config.get("cache_dir", False):
                    raise ValueError("cache_dir must be specified in config when using lmdb cache_mode.")
                return LMDBDataset(
                    data=data_list,
                    transform=transforms,
                    cache_dir=dataset_config.cache_dir,
                    hash_transform=pickle_hashing
                )
            case _:
                return Dataset(
                    data=data_list,
                    transform=transforms,
                )
