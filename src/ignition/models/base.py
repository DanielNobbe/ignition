from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import nn


class IgnitionModel(nn.Module, ABC):
    """Base class for models in Ignition.
    Note that as of now, this class is used for generic models,

    Ignition also supports loading MONAI models directly from the config.
    This class can be used for custom models, but MONAI-style models
    can also be loaded directly from the config.
    """
    # name might conflict with other "model", e.g. data model
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x):
        """Forward pass of the model."""
        return self.get_model()(x)
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Returns the model instance."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def get_parameters(self) -> nn.Parameter:
        """Returns the parameters of the model, for initializing the optimizer."""
        return self.get_model().parameters()

    @abstractmethod
    def get_model_transform(self) -> Callable:
        """Returns a function that transforms the model output to the format expected by Ignite."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def get_model_transform(self) -> Callable:
        """Returns a function that transforms the model output for training. The returned function has 
        inputs:  x, y, y_pred, loss
        outputs: {'y_pred': y_pred, 'y': y, 'train_loss': loss.item()} (this may be dependent on the task)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def get_train_values_output_transform(self):
        """Specify how to get the predictions and targets from the output from the train_model_output_transform.
        Needed to recompute the loss for logging.
        The returned function has
        inputs: {'y_pred': y_pred, 'y': y, 'train_loss': loss.item()}
        outputs: (y_pred, y)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    @abstractmethod
    def get_eval_output_transform(self) -> Callable:
        """Returns a function that transforms the model output for evaluation.
        The returned function has
        inputs: x, y, raw_model_output
        outputs: (y_pred, y)"""
        raise NotImplementedError("This method should be implemented by subclasses.")
