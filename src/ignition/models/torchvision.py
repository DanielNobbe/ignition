from .base import IgnitionModel

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

from ignite.utils import convert_tensor


class TorchVisionSegmentationModel(IgnitionModel):
    # Maybe better to use the get_model method to return the nn.Module wherever it's needed?
    def __init__(self, config):
        super().__init__(config)
        self.model = self.setup_model()

    def setup_model(self):
        if self.config.model.name == "deeplabv3_mobilenet_v3_large":
            if self.config.model.pretrained:
                return deeplabv3_mobilenet_v3_large(
                    num_classes=self.config.num_classes, weights="DEFAULT", weights_backbone="DEFAULT"
                )
            else:
                return deeplabv3_mobilenet_v3_large(
                    num_classes=self.config.num_classes, weights=None, weights_backbone=None
                )
        elif self.config.model.name == "deeplabv3_resnet50":
            if self.config.model.pretrained:
                return deeplabv3_resnet50(num_classes=self.config.num_classes, weights="DEFAULT")
            else:
                return deeplabv3_resnet50(num_classes=self.config.num_classes, weights=None)
        else:
            raise ValueError(f"Model {self.config.model.name} is not supported.")

    def get_model(self):
        return self.model
    
    def get_parameters(self):
        """Returns the parameters of the model, for initializing the optimizer."""
        return self.model.parameters()
    
    def get_model_transform(self):
        def model_transform(output):
            """Transform model output for training.
            For some reason, the model returns a dict with 'out' key."""
            return output['out']
        
        return model_transform

    def get_train_output_transform(self):
        def train_output_transform(_, y, y_pred, loss):
            """Transform model output for evaluation.
            inputs: x, y, y_pred, loss
            """
            return {
                'y_pred': y_pred, 
                'y': y,
                'train_loss': loss.item()
            }
        
        return train_output_transform
    
    def get_train_values_output_transform(self):
        """Specify how to get the predictions and targets from the output from the train_model_output_transform.
        Needed to recompute the loss for logging."""
        def train_loss_output_transform(output):
            """Transform model output for training."""
            return output['y_pred'], output['y']
        
        return train_loss_output_transform

    def get_eval_output_transform(self):
        def eval_output_transform(_, y, y_pred):
            """Transform model output for evaluation.
            For some reason, the model returns a dict with 'out' key.
            For some reason, this needs to return a tuple for the evaluator to work correctly.
            inputs: x, y, y_pred
            """
            return y_pred['out'], y
        
        return eval_output_transform