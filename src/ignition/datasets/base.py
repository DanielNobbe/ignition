

"""
We want a dataset class, or something with another name, that
can do the following:


- set up (and hold) train and validation datasets
- or, only one of the two
- return a dataloader for each dataset

internal functions:
- set up the datasets
- set up transforms
- set up the dataloaders
- allow subsetting of the datasets


-- > this class needs some more work, but not sure
which functions are necessary in a generic setup.
Maybe we could specify transforms in e.g. the config.
"""

import torch
from warnings import warn
from abc import ABC, abstractmethod

"""
We first make a class that can create and hold a single
dataset, and return a dataloader for it.

"""
class IgnitionDataset(ABC):
    def __init__(self, config, name='train'):
        self.name = name
        # if name in config, use it, otherwise assume there
        # is only one dataset
        self.config = config.dataset[name]
        # this gives an error if the name is not found, which is fine
        self.dataset = self._setup_dataset()
        self.dataloader = self._setup_dataloader()

    @abstractmethod
    def _setup_dataset(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @abstractmethod
    def _setup_dataloader(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    def get_dataloader(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataset)
    

"""
A dataset class that can hold both train and validation datasets,
and return dataloaders for each.

Note: We do not need to implement creating
each individual dataset if we create the paired dataset directly.

"""
class PairedDataset(ABC):
    def __init__(self, config):
        # assert len(config.dataset) == 2, "Config must contain exactly two datasets, a train and eval set. They can have any name."
        self.config = config
        self._setup_datasets()

    def _find_train_key(self):
        for key in self.config.dataset.keys():
            if 'train' in key.lower():
                return key
        # otherwise, return the first key
        warn("No 'train' key found in dataset config, using the first key.")
        return list(self.config.dataset.keys())[0]
    
    def _find_eval_key(self):
        for key in self.config.dataset.keys():
            if 'eval' in key.lower() or 'val' in key.lower():
                return key
        # otherwise, return the first key
        warn("No 'eval' or 'val' key found in dataset config, using the second key.")
        return list(self.config.dataset.keys())[1]
    

    @abstractmethod
    def get_train_dataloader(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @abstractmethod
    def get_eval_dataloader(self):
        raise NotImplementedError("This method should be implemented in a subclass")
    
    @abstractmethod
    def _setup_datasets(self):
        """This method should be implemented in subclasses to set up the datasets."""
        raise NotImplementedError("This method should be implemented in a subclass")

    @abstractmethod
    def get_prepare_batch(self):
        """Returns a function that prepares the batch for the model."""
        raise NotImplementedError("This method should be implemented in a subclass")
