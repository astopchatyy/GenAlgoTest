import torch
from typing import List
from .metric import Metric


class MSE(Metric):
    def eval(self, predicted_values:torch.Tensor, true_values:torch.Tensor) -> float:
        return torch.mean((predicted_values - true_values) ** 2).item()

    def aggregate(self, values: List) -> float:
        return sum(values)