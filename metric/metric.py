import torch
from typing import List
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, predicted_values: torch.Tensor, true_values: torch.Tensor) -> float:
        return 0.0

    @abstractmethod
    def aggregate(self, values: List) -> float:
        return 0.0