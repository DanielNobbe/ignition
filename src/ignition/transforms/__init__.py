import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.transforms import MapLabelValue
from monai.transforms.transform import Transform, MapTransform

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
        if self.remove_old_keys:
            new_data = {}
        else:
            new_data = d.copy()
        for old_key, new_key in self.key_mapping.items():
            if old_key in d:
                new_data[new_key] = d[old_key]
                if not self.remove_old_keys and new_key != old_key:
                    # still need to remove the previous key so there's no remaining old key
                    del new_data[old_key]
        return new_data
        