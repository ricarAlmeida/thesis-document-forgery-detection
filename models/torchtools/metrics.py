""" Metrics Definition """

from typing import Dict, Optional
from abc import ABC, abstractmethod
import numpy as np

import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics import AUROC, JaccardIndex, Specificity
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Metric(ABC):
    """
    Metric abstraction
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Resets metric tracking
        """
        
        pass

    @abstractmethod
    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Tracks the metric of a given batch of predictions and ground-truth labels

        Parameters:
            predictions: model predictions/logits
            labels: ground-truth labels
        """
        
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """
        Computes the average tracked metric and resets the metric tracking
        """
        
        pass

    @abstractmethod
    def evaluate_without_reset(self) -> float:
        """
        Computes the average tracked metric without reseting the metric tracking
        """
        
        pass

    @abstractmethod
    def compute(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Computes the metric for a given batch of predictions and ground-truth labels

        Parameters:
            predictions: model predictions/logits
            labels: ground-truth labels
        """
        
        pass
        
    @abstractmethod
    def to(self, device: torch.device):
        """
        Put metric on the given device
        """
        
        pass
        

class BinaryPrecisionMetric(Metric):
    """
    Precision for binary tasks
    
    Note: if predictions outside [0, 1] range, they are automatically considered logits

    Attributes:
        threshold: minimum value to be considered as positive
        ignore_index: index ignored for metric calculation
    """

    def __init__(self, threshold: float = 0.5, ignore_index:int = -1):
        self.metric = BinaryPrecision(threshold=threshold, ignore_index=ignore_index)
        self.precisions = []

    def reset(self) -> None:
        self.precisions = []

    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        precision = self.metric(predictions, labels)
        self.precisions.append(precision.item())
        return precision.item()

    def evaluate(self) -> float:
        x = np.array(self.precisions).mean()
        self.reset()
        return x

    def evaluate_without_reset(self) -> float:
        return np.array(self.precisions).mean()

    def compute(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        return self.metric(predictions, labels).item()

    def to(self, device: torch.device):
        self.metric.to(device)


class BinaryRecallMetric(Metric):
    """
    Recall for binary tasks

    Note: if predictions outside [0, 1] range, they are automatically considered logits

    Attributes:
        threshold: minimum value to be considered as positive
        ignore_index: index ignored for metric calculation
    """

    def __init__(self, threshold: float = 0.5, ignore_index:int = -1):
        self.metric = BinaryRecall(threshold=threshold, ignore_index=ignore_index)
        self.recalls = []

    def reset(self) -> None:
        self.recalls = []

    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        recall = self.metric(predictions, labels)
        self.recalls.append(recall.item())
        return recall.item()

    def evaluate(self) -> float:
        x = np.array(self.recalls).mean()
        self.reset()
        return x

    def evaluate_without_reset(self) -> float:
        return np.array(self.recalls).mean()

    def compute(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        return self.metric(predictions, labels).item()

    def to(self, device: torch.device):
        self.metric.to(device)
        

def f1_score(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall + 1e-8)


class BinaryF1ScoreMetric(Metric):
    """
    F1 score for binary tasks

    Note: if predictions outside [0, 1] range, they are automatically considered logits

    Attributes:
        threshold: minimum value to be considered as positive
        ignore_index: index ignored for metric calculation
    """

    def __init__(self, threshold: float = 0.5, ignore_index: int = -1):
        self.precision_metric = BinaryPrecisionMetric(threshold=threshold, ignore_index=ignore_index)
        self.recall_metric = BinaryRecallMetric(threshold=threshold, ignore_index=ignore_index)

    def reset(self) -> None:
        self.precision_metric.reset()
        self.recall_metric.reset()

    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        batch_precision = self.precision_metric.add_batch(predictions, labels)
        batch_recall = self.recall_metric.add_batch(predictions, labels)
        return f1_score(batch_precision, batch_recall)

    def evaluate(self) -> float:
        mean_precision = self.precision_metric.evaluate_without_reset()
        mean_recall = self.recall_metric.evaluate_without_reset()
        self.reset()
        return f1_score(mean_precision, mean_recall)

    def evaluate_without_reset(self) -> float:
        mean_precision = self.precision_metric.evaluate_without_reset()
        mean_recall = self.recall_metric.evaluate_without_reset()
        return f1_score(mean_precision, mean_recall)

    def evaluate_precision(self) -> float:
        return self.precision_metric.evaluate_without_reset()

    def evaluate_recall(self) -> float:
        return self.recall_metric.evaluate_without_reset()

    def compute(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        batch_precision = self.precision_metric.compute(predictions, labels)
        batch_recall = self.recall_metric.compute(predictions, labels)
        return f1_score(batch_precision, batch_recall)
    
    def compute_precision(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        return self.precision_metric.compute(predictions, labels)

    def compute_recall(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        return self.recall_metric.compute(predictions, labels)

    def to(self, device: torch.device):
        self.precision_metric.to(device)
        self.recall_metric.to(device)


class Metrics:
    def __init__(self, metrics: Optional[Dict[str, Metric]]):
        self.metrics = metrics if metrics is not None else {}

    def reset(self):
        for _, v in self.metrics.items():
            v.reset()
    
    def add_batch(self, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        values = {}
        for name, v in self.metrics.items():
            values[name] = v.add_batch(predictions, labels)

        return values

    def evaluate(self) -> Dict[str, float]:
        evaluations = {
            k: v.evaluate_without_reset()
            for k, v in self.metrics.items()
        }
        self.reset()
        return evaluations

    def to(self, device: torch.device):
        for key, metric in self.metrics.items():
            metric.to(device)