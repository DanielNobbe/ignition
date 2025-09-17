from monai.transforms import Transform
from monai.config.type_definitions import KeysCollection

class DSGetFirst(Transform):
    """Transform to get the first element from a list or tuple.

    This is useful for handling outputs from deep supervision models,
    where the output is a list of tensors. This transform extracts
    the first tensor from the list.

    Args:
        None

    """

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            return data[0]
        else:
            raise TypeError("Input must be a list or tuple.")
        return data
    
class DSGetFirstd(Transform):
    """Dictionary-based transform to get the first element from a list or tuple."""
    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: Keys of the corresponding items to be transformed.
        """
        self.keys = keys
        self.converter = DSGetFirst()

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d and isinstance(d[key], (list, tuple)):
                d[key] = self.converter(d[key])
        return d