import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, List, Type, Any
from tqdm import tqdm
import math
from metric import Metric

class BasePopulationTrainer:
    def __init__(self,
                 population_size: int,
                 model_class: Type[nn.Module],
                 optimizer_class: Type[optim.Optimizer],
                 criterion: nn.modules.loss._Loss,
                 metric: Metric,
                 model_params: Dict={},
                 optimizer_params: Dict={}) -> None:

        self._model_class = model_class
        self._optimizer_class = optimizer_class
        self._total_population_size = population_size
        self._metric = metric
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model_params = model_params
        self._optimizer_params = optimizer_params

        self._population = [self._model_class(**self._model_params).to(self._device) 
                            for _ in range(self._total_population_size)]
        self._criterion = criterion
        self._optimizers = [self._optimizer_class(self._population[i].parameters(), **self._optimizer_params) 
                            for i in range(self._total_population_size)]

        self._training_history = None

    def _train_one_model_one_epoch(self,
                                   model: nn.Module,
                                   optimizer: optim.Optimizer,
                                   criterion: nn.modules.loss._Loss,
                                   dataloader: DataLoader,
                                   pbar: tqdm|None=None,
                                   pbar_message: str|None=None) -> List:
        model.train()
        train_losses = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            if pbar:
                if pbar_message:
                    pbar.set_description(f"{pbar_message}{i+1}/{len(dataloader)}")
                pbar.update(1)

        return train_losses

    def _evaluate_model(self, model: nn.Module, dataloader: DataLoader, pbar: tqdm|None=None, pbar_message: str|None=None) -> float:
        model.eval()
        scores = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                outputs = model(inputs)
                scores.append(self._metric.eval(outputs, labels)) 
                if pbar:
                    if pbar_message:
                        pbar.set_description(f"{pbar_message}{i+1}/{len(dataloader)}")
                    pbar.update(1)
        return self._metric.aggregate(scores)

    def _reset_optimizer(self, optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
        param_groups = optimizer.param_groups
        base_hyperparams = {k: v for k, v in param_groups[0].items() if k != 'params'}
        return self._optimizer_class(model.parameters(), **base_hyperparams)
    
    
    def _update_population(self, 
                           cycle: int, 
                           cycles: int, 
                           fitness_scores: List[float], 
                           **kwargs: Any) -> List:

        return [None for i in range(self._total_population_size)]


    def train(self,
              cycles: int,
              train_dataloader: DataLoader,
              test_dataloader: DataLoader,
              validation_dataloader: DataLoader|None=None,
              epoches_per_cycle: int = 1,
              verbose: int = 2,
              **kwargs: Any) -> Dict: # Додано **kwargs

        if not validation_dataloader:
            validation_dataloader = train_dataloader

        self._training_history = {
            "update_history": [[] for _ in range(self._total_population_size)],
            "train_losses": [[] for _ in range(self._total_population_size)],
            "test_metric": [[] for _ in range(self._total_population_size)],
            "val_metric": [[] for _ in range(self._total_population_size)]
        }
        
        if verbose == 2:
            total_len = cycles * self._total_population_size * epoches_per_cycle * (len(train_dataloader) + len(validation_dataloader) + len(test_dataloader))
        elif verbose == 1:
            total_len = cycles * self._total_population_size * epoches_per_cycle
        else:
            total_len = cycles
            
        with tqdm(total=total_len, leave=False) as pbar:
            for cycle in range(cycles):
                if verbose not in [1, 2]:
                    pbar.set_description(f"Cycle: {cycle+1}/{cycles}")
                    pbar.update(1)
                    
                fitness_scores = []
                for i in range(self._total_population_size):
                    fitness_list = []
                    test_fitness_list = []
                    train_losses_list = []

                    for epoch in range(epoches_per_cycle):
                        message = f"Cycle: {cycle+1}/{cycles}, entity: {i+1}/{self._total_population_size}, epoch: {epoch+1}/{epoches_per_cycle}"
                        
                        if verbose == 2:
                            progr_bar = pbar
                        else:
                            progr_bar = None
                            if verbose == 1:
                                pbar.set_description(message)
                                pbar.update(1)
                                
                        losses = self._train_one_model_one_epoch(self._population[i],
                                                                 self._optimizers[i],
                                                                 self._criterion,
                                                                 train_dataloader,
                                                                 progr_bar,
                                                                 f"{message}, training: ")
                        train_losses_list.append(losses)

                        fitness = self._evaluate_model(self._population[i], validation_dataloader, progr_bar, f"{message}, evaluating validation: ")
                        fitness_list.append(fitness)

                        test_fitness = self._evaluate_model(self._population[i], test_dataloader, progr_bar, f"{message}, evaluating test: ")
                        test_fitness_list.append(test_fitness)

                    self._training_history["train_losses"][i].append(train_losses_list)
                    self._training_history["val_metric"][i].append(fitness_list)
                    self._training_history["test_metric"][i].append(test_fitness_list)
                
                update_history = self._update_population(cycle, cycles, fitness_scores, **kwargs)
                
                for i in range(self._total_population_size):
                    self._training_history["update_history"][i].append(update_history[i])

        return self._training_history

    def extract_best_model(self) -> nn.Module:
        if not self._training_history:
            raise ValueError("Training not performed yet.")

        final_val_metrics = [
            np.mean(model_metrics[-1]) if model_metrics and model_metrics[-1] else 0.0
            for model_metrics in self._training_history["val_metric"]
        ]

        best_model_index = int(np.argmax(final_val_metrics))
        
        return self._population[best_model_index]