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
from trainer import BasePopulationTrainer

class GeneticTrainer(BasePopulationTrainer):
    def __init__(self,
                 genetic_population_size: int,
                 superviser_population_size: int,
                 model_class: Type[nn.Module],
                 optimizer_class: Type[optim.Optimizer],
                 criterion: nn.modules.loss._Loss,
                 metric: Metric,
                 model_params: Dict={},
                 optimizer_params: Dict={}) -> None:

        total_population_size = genetic_population_size + superviser_population_size
        
        super().__init__(
            population_size=total_population_size,
            model_class=model_class,
            optimizer_class=optimizer_class,
            criterion=criterion,
            metric=metric,
            model_params=model_params,
            optimizer_params=optimizer_params
        )
        
        self._genetic_population_size = genetic_population_size
        self._superviser_population_size = superviser_population_size

    def _update_population(self, 
                            cycle: int, 
                            cycles: int, 
                            fitness_scores: List[float],
                            survivor_fraction: float = 0.5,
                            mutation_chance: float = 0.1,
                            last_cycle_evolution=True, 
                            **kwargs: Any) -> List:
        
        history: List[Any] = [None for i in range(self._total_population_size)]
        
        if cycle < cycles - 1 or last_cycle_evolution:
            ranked = sorted(list(zip(fitness_scores[:self._genetic_population_size], list(range(self._genetic_population_size)))), key=lambda x: x[0], reverse=True)

            survivors_count = math.ceil(self._genetic_population_size * survivor_fraction)

            for _, index in ranked[survivors_count:]:
                index_a, index_b = random.sample(range(survivors_count), 2)
                history[index] = (cycle, index_a, index_b)

                self._population[index] = self._crossover(self._population[index_a], self._population[index_b], mutation_chance)
                self._optimizers[index] = self._reset_optimizer(self._optimizers[index], self._population[index])


        return history


    def _crossover(self, model_a: nn.Module, model_b: nn.Module, mutation_chance: float) -> nn.Module:
        child = self._model_class(**self._model_params).to(self._device)
        with torch.no_grad():
            mods_a = dict(model_a.named_modules())
            mods_b = dict(model_b.named_modules())
            if self._superviser_population_size and mutation_chance:
                mutation_index = random.sample(range(self._superviser_population_size), 1)[0]
                mods_mut = dict(self._population[mutation_index].named_modules())

            for name, module_child in child.named_modules():

                if len(list(module_child.children())) != 0:
                    continue

                if self._superviser_population_size and  mutation_chance and random.random() < mutation_chance:
                    src = mods_mut[name] # type: ignore
                elif random.random() < 0.5:
                    src = mods_a[name]
                else:
                    src = mods_b[name]

                for param_child, param_src in zip(module_child.parameters(), src.parameters()):
                    param_child.data.copy_(param_src.data)

                for buf_name, buf_child in module_child.named_buffers():
                    buf_src = getattr(src, buf_name)
                    buf_child.copy_(buf_src)
                    return child
        return child

    def _layerwise_mse(self, model_a: nn.Module, model_b: nn.Module):
        layers_a = [m for m in model_a.modules() if len(list(m.parameters())) > 0]
        layers_b = [m for m in model_b.modules() if len(list(m.parameters())) > 0]

        mses = []
        for la, lb in zip(layers_a, layers_b):
            params_a = torch.cat([p.view(-1) for p in la.parameters()])
            params_b = torch.cat([p.view(-1) for p in lb.parameters()])
            mse = nn.functional.mse_loss(params_a, params_b).item()
            mses.append(mse)

        return torch.mean(torch.Tensor(mses)).item()

    def extract_model(self, smilarity_fraction: float=0.8) -> nn.Module:
        mses = [[] for i in range(self._genetic_population_size)]
        for i in range(self._genetic_population_size):
          for j in range(i + 1, self._genetic_population_size):
            mses[i].append(self._layerwise_mse(self._population[i], self._population[j]))
            mses[j].append(self._layerwise_mse(self._population[i], self._population[j]))

        row_means = np.mean(mses, axis=1)
        selector = row_means <= np.quantile(row_means, smilarity_fraction)

        avg_model = self._model_class(**self._model_params).to(self._device)
        avg_state = avg_model.state_dict()

        keys = avg_state.keys()
        for key in keys:
            stacked = torch.stack([self._population[i].state_dict()[key] for i in range(self._genetic_population_size) if selector[i]], dim=0)
            avg_state[key] = stacked.mean(dim=0)

        avg_model.load_state_dict(avg_state)
        return avg_model
