import os
from collections.abc import Iterable

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.transforms import MapLabelValue
from monai.transforms.transform import Transform, MapTransform, RandomizableTrait

from monai.utils import look_up_option



class VistaLabelMapd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        label_mapping: list[tuple[int, int]],
        dtype: DtypeLike = np.int16,
        from_onehot: bool = False,
        invert: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Remap labels using Vista3D-style tuple-based label map.

        Args:
            keys: keys of the corresponding items to be transformed.
            label_mapping: list of tuples specifying the mapping from original to target labels.
                For example, [(0, 0), (1, 10), (2, 20)] maps label 0 to 0, label 1 to 10, and label 2 to 20.
            dtype: data type of the output label map.
            from_onehot: if True, assumes input labels are 0-indexed consecutive integers
                (e.g., from one-hot encoding) rather than the original label values.
                Default is False.
            invert: if True, inverts the mapping direction. Default is False.
        """
        super().__init__(keys, allow_missing_keys)

        source_label_index_in_tuple = 1 if invert else 0
        target_label_index_in_tuple = 0 if invert else 1

        if from_onehot:
            # we will have a 0-indexed consecutive integer label input,
            # rather than the original label values.
            # so we need to use the ordering in the mapping to determine the 
            # indices
            self.mapper = MapLabelValue(
                orig_labels=list(range(len(label_mapping))),
                target_labels=[int(tup[target_label_index_in_tuple]) for tup in label_mapping],
                dtype=dtype
            )
        else:
            self.mapper = MapLabelValue(
                    orig_labels=[int(tup[source_label_index_in_tuple]) for tup in label_mapping],
                    target_labels=[int(tup[target_label_index_in_tuple]) for tup in label_mapping],
                    dtype=dtype
                )

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.mapper(d[key])
        return d
    

class RiCheckLabelMap(Transform):
    def __init__(
        self,
        expected_label_map: dict[int, str],
        label_map_key: str = "label_map",
    ) -> None:
        """
        Check if the label map contains the expected labels.

        Args:
            keys: keys of the corresponding items to be transformed.
            expected_labels: list of expected label values.
        """
        self.expected_label_map = expected_label_map
        self.label_map_key = label_map_key

    def __call__(self, data):
        if self.label_map_key not in data:
            raise KeyError(f"Label map key '{self.label_map_key}' not found in data.")
        label_map = data[self.label_map_key]
        if not label_map == self.expected_label_map:
            raise ValueError(
                f"Label map does not match expected. "
                f"Expected: {self.expected_label_map}, Got: {label_map}"
            )
        return data
        
class MapKeysTransform(Transform):
    def __init__(
        self,
        key_mapping: dict[str, str],
        remove_old_keys: bool = True,
    ) -> None:
        """
        Map keys in a dictionary to new keys.

        Args:
            key_mapping: dictionary specifying the mapping from old keys to new keys.
            remove_old_keys: if True, removes the old keys from the dictionary after mapping. Default is True.
        """
        self.key_mapping = key_mapping
        self.remove_old_keys = remove_old_keys

    def __call__(self, data):
        d = dict(data)
        new_data = d.copy()
        for old_key, new_key in self.key_mapping.items():
            if old_key in d:
                new_data[new_key] = d[old_key]
                if self.remove_old_keys and new_key != old_key:
                    # still need to remove the previous key so there's no remaining old key
                    del new_data[old_key]
        return new_data
        
class MultipleItemRandomSelect(Transform, RandomizableTrait):
    """Used for datasets with multiple files per image. Randomly selects one of them.
    
    Should be used in conjunction with MultipleItemRandomSelectd.
    """

    def __init__(
        self,
        mode: str = "uniform"
    ):
        """
        Args:
            mode: selection mode. Options are "uniform" (default) or "weighted".
        """
        assert mode == "uniform", f"Unsupported mode: {mode}"
        self.mode = mode

    def __call__(self, data):
        if not isinstance(data, list):
            raise ValueError("Input data must be a list.")
        if self.mode == "uniform":
            idx = np.random.randint(len(data))
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
        return data[idx]


class MultipleItemRandomSelectd(MapTransform, RandomizableTrait):
    """Used for datasets with multiple files per image. Randomly selects one of them.

    Should be used in conjunction with MultipleItemRandomSelectd, and before LoadImaged.
    """

    def __init__(
        self,
        keys: KeysCollection,
        mode: str = "uniform",
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            mode: selection mode. Options are "uniform" (default) or "weighted".
        """
        super().__init__(keys, allow_missing_keys)
        assert mode == "uniform", f"Unsupported mode: {mode}"
        self.mode = mode
        self.transform = MultipleItemRandomSelect(mode=mode)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class MultipleItemSelectFirst(Transform):
    """Used for datasets with multiple files per image. Selects the first one, or a specified index.
    
    If input is not a list, returns the input as is.
    """

    def __init__(
        self,
        index: int = 0,
        optional: bool = True,

    ):
        """
        Args:
            mode: selection mode. Options are "uniform" (default) or "weighted".
        """
        self.index = index
        self.optional = optional

    def __call__(self, data):
        if self.optional and not isinstance(data, list):  # Iterable fails because str is Iterable
            return data
        # breakpoint()
        return data[self.index]


class MultipleItemSelectFirstd(MapTransform):
    """Used for datasets with multiple files per image. Selects the first one.

    Should be used in conjunction with MultipleItemRandomSelectd, and before LoadImaged.
    """

    def __init__(
        self,
        keys: KeysCollection,
        index: int = 0,
        optional: bool = True,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            mode: selection mode. Options are "uniform" (default) or "weighted".
        """
        super().__init__(keys, allow_missing_keys)
        self.transform = MultipleItemSelectFirst(index=index, optional=optional)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class GetKey(Transform):
    """Get a specific key from a dictionary."""

    def __init__(
        self,
        key: str,
    ):
        """
        Args:
            key: key to get from the dictionary.
        """
        self.key = key

    def __call__(self, data):
        if isinstance(data, dict):
            if self.key not in data:
                raise KeyError(f"Key '{self.key}' not found in data.")
            return data[self.key]
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return [item[self.key] for item in data if self.key in item]
        else:
            raise TypeError("Input data must be a dictionary or a list of dictionaries.")


class GetKeyd(MapTransform):
    """Get a specific key from a dictionary."""

    def __init__(
        self,
        keys: KeysCollection,
        key: str,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: Keys in item to apply transform to
            key: key to get from the dictionary.

        example:
        item = {
            'label': {
                'masks: ...
                'annotator': 'Dr. Smith'
            },
            'image': ...
        }
        keys = 'label'
        key = 'masks'

        --> this transform will extract item['label']['masks'] to item['label']

        """
        self.transform = GetKey(key)
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class MakeAbsPathd(MapTransform):
    """Make paths absolute by joining with a root directory."""

    def __init__(
        self,
        keys: KeysCollection,
        root_dir: str,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            root_dir: root directory to join with relative paths.
        """
        super().__init__(keys, allow_missing_keys)
        self.root_dir = root_dir

    def __call__(self, data):

        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = os.path.abspath(os.path.join(self.root_dir, d[key]))
        return d
    

class SqueezeDim(Transform):
    """Squeeze a specific dimension from a tensor or numpy array."""

    def __init__(
        self,
        dim: int = -1,
    ):
        """
        Args:
            dim: dimension to squeeze. Default is the last.
        """
        self.dim = dim

    def __call__(self, data):
        if isinstance(data, torch.Tensor | np.ndarray):
            return data.squeeze(self.dim)
        else:
            raise TypeError("Input data must be a torch.Tensor or a numpy.ndarray.")
        
class SqueezeDimd(MapTransform):
    """Squeeze a specific dimension from a tensor or numpy array."""

    def __init__(
        self,
        keys: KeysCollection,
        dim: int = -1,
        allow_missing_keys: bool = False,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            dim: dimension to squeeze. Default is the last.
        """
        super().__init__(keys, allow_missing_keys)
        self.transform = SqueezeDim(dim=dim)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d
    

class Debugd(MapTransform):
    """Print the shape and type of the data for debugging purposes."""

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = True,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
        """
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        breakpoint()

        return d