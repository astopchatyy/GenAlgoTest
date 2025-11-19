import torch
from typing import List
from .metric import Metric


class Accuracy(Metric):
    def eval(self, predicted_values:torch.Tensor, true_values:torch.Tensor) -> float:
        _, predicted = torch.max(predicted_values, 1)
        return (predicted == true_values).float().mean().item()

    def aggregate(self, values: List) -> float:
        return sum(values) / len(values)