# NOTE: COuld be in __init__.py, but keeping it here for clarity


"""
Each model should be a class, inheriting from AbstractModel, with the following methods:

- get_model: returns the model in a format expected by Ignite
    - a torch.nn.Module that can be trained using a typical Pytorch training loop
- get_parameters: returns the parameters of the model, for initializing the optimizer
- get_model_transform: returns a function that transforms the model output to the format expected by Ignite
    --> This ensures the output is a tensor of values, i.e. the pure model output
- get_train_output_transform: 
        "function that receives 'x', 'y', 'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`."
    --> This sets the trainers internal state, so we can do whatever makes sense.
    --> Current implementation relies on having this dict: {'y_pred': y_pred, 'y': y,'train_loss': loss.item()}
- get_eval_output_transform: 
        "function that receives 'x', 'y', 'y_pred' and returns value
            to be assigned to engine's state.output after each iteration. Default is returning `(y_pred, y,)` which fits
            output expected by metrics. If you change it you should use `output_transform` in metrics."
        --> So here it makes sense to keep the tuple output, as it is expected by the metrics.

maybe we also add:
- get_loss_function: returns the loss function to be used for training
    --> this removes some flexibility, but typically a model arch depends on a certain loss fn
    --> probably best to have them separately

"""

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
    

    def get_model_transform(self) -> Callable:
        """Returns a function that transforms the model output to the format expected by Ignite."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_train_output_transform(self) -> Callable:
        """Returns a function that transforms the model output for training."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_train_values_output_transform(self):
        """Specify how to get the predictions and targets from the output from the train_model_output_transform.
        Needed to recompute the loss for logging."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_eval_output_transform(self) -> Callable:
        """Returns a function that transforms the model output for evaluation."""
        raise NotImplementedError("This method should be implemented by subclasses.")
