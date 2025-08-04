from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large


def setup_model(config):
    """Setup model based on the configuration."""
    if config.model.name == "deeplabv3_mobilenet_v3_large":
        if config.model.pretrained:
            return deeplabv3_mobilenet_v3_large(
                num_classes=config.num_classes, weights="DEFAULT", weights_backbone="DEFAULT"
            )
        else:
            return deeplabv3_mobilenet_v3_large(
                num_classes=config.num_classes, weights=None, weights_backbone=None
            )
    elif config.model.name == "deeplabv3_resnet50":
        if config.model.pretrained:
            return deeplabv3_resnet50(num_classes=config.num_classes, weights="DEFAULT")
        else:
            return deeplabv3_resnet50(num_classes=config.num_classes, weights=None)
