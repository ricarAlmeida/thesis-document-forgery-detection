""" Collection of learning rate schedulers """

from typing import Optional
from enum import Enum, auto
from abc import ABC, abstractmethod


class SchedulerReference(Enum):
    """
    Scheduler reference, between epoch or iteration/batch
    """

    ITERATION = auto()
    EPOCH = auto()


class Scheduler(ABC):
    """
    Abstraction of a learning rate scheduler
    """

    @property
    @abstractmethod
    def reference(self) -> SchedulerReference:
        """
        Retrieves the scheduler reference
        """
        
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """
        Retrieves a learning rate value
        """
        
        pass


class PowerDecayScheduler(Scheduler):
    """
    Power Decay Scheduler based on iterations.
    Consider an iteration as a single batch

    Attributes:
        lr_0: initial learning rate
        max_iters: maximum number of iterations
        power: exponent value
    """

    def __init__(self, lr_0: float, max_iters: Optional[int], power: float):
        self.lr = lr_0
        self.max_iters = max_iters
        self.power = power

    @property
    def reference(self) -> SchedulerReference:
        return SchedulerReference.ITERATION
    
    def __call__(self, iteration: int, **kwargs) -> float:
        """
        Parameters:
            cur_iters: current number of iterations/batches
        """
        
        if "max_iters" in kwargs:
            self.max_iters = kwargs["max_iters"]
        return self.lr*((1-float(iteration)/self.max_iters)**(self.power))


class CosineScheduler(Scheduler):

    """ 
    Cosine learning rate decay with warmup

    Attributes:
        max_epoch: epoch number where learning rate reaches its minimum
        lr_0: initial learning rate
        lr_min: minimum possible learning rate 
        warmup_steps: number of warmup steps, period where the learning rate increases linearly
        warmup_begin_lr: warmup initial learning rate
    
    """
    
    def __init__(
        self,
        max_epoch: float,
        lr_0: float, 
        lr_min: float,
        warmup_steps: int = 0, 
        warmup_begin_lr: int = 0,
    ):

        assert warmup_begin_lr < lr_0
        assert lr_min < lr_0
        
        self.lr_0 = lr_0
        self.max_epoch = max_epoch
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_epoch - self.warmup_steps

    @property
    def reference(self) -> SchedulerReference:
        return SchedulerReference.EPOCH

    def get_warmup_lr(self, epoch: int):
        increase = (self.lr_0 - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch: int, **kwargs) -> float:
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
            
        if epoch <= self.max_epoch:
            self.lr = self.lr_min + (
                self.lr_0 - self.lr_min
            ) * (
                1 + math.cos(
                    math.pi * (epoch - self.warmup_steps) / self.max_steps
                )
            ) / 2
            
        return self.lr