""" Abstraction of a PyTorch model """

from typing import Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


@dataclass
class ModelOutput:

    logits: torch.Tensor
    labels: Optional[torch.Tensor]
    loss: Optional[torch.Tensor]
    

class ModelRunner(ABC):

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """
        Attribute of a pytorch model of type nn.Module
        """
        
        pass

    @abstractmethod
    def logits(self, batch: dict, device: str, loss_fn: Optional[_Loss]) -> ModelOutput:
        """
        Run the model for a given batch using the specified device (CPU or GPU)

        Parameters:
            batch: batch of examples
            device: cpu or gpu
            loss_fn: loss function to optimize

        Returns: 
            tuple with the loss and logits 
        """
        
        pass

        