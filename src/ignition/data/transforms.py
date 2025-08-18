import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


class IgnoreMaskBoundaries(A.DualTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        # If no change to image, just pass it through
        return img

    def apply_to_mask(self, mask, **params):
        mask = mask.copy()
        mask[mask == 255] = 0
        return mask
