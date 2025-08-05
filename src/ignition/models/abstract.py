import torch
from torch import nn

from typing import Callable

class IgnitionModel(nn.Module):
    # name might conflict with other "model", e.g. data model
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        """Forward pass of the model."""
        return self.get_model()(x)
    
    def get_model(self) -> nn.Module:
        """Returns the model instance."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_parameters(self) -> nn.Parameter:
        """Returns the parameters of the model, for initializing the optimizer."""
        return self.get_model().parameters()

    # NOTE: Could also specify these functions as properties, which would allow for us to check their inputs/outputs. This would not allow to use the config to set the function, but the function would still have access to the config. In the end it comes down to a difference in memory, but that's not a big deal.
    # only the get_model_transform does not always have the same input signature. So for uniformity, we keep all as returning a method.
    def get_model_transform(self) -> Callable:
        """Returns a function that transforms the model output to the format expected by Ignite."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_model_transform(self) -> Callable:
        """Returns a function that transforms the model output for training. The returned function has 
        inputs:  x, y, y_pred, loss
        outputs: {'y_pred': y_pred, 'y': y, 'train_loss': loss.item()} (this may be dependent on the task)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_train_values_output_transform(self):
        """Specify how to get the predictions and targets from the output from the train_model_output_transform.
        Needed to recompute the loss for logging.
        The returned function has
        inputs: {'y_pred': y_pred, 'y': y, 'train_loss': loss.item()}
        outputs: (y_pred, y)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_eval_output_transform(self) -> Callable:
        """Returns a function that transforms the model output for evaluation.
        The returned function has
        inputs: x, y, raw_model_output
        outputs: (y_pred, y)"""
        raise NotImplementedError("This method should be implemented by subclasses.")
