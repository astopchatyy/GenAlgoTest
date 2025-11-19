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
from .basic_trainer import BasePopulationTrainer


class ShuffleTrainer(BasePopulationTrainer):
    def __init__(self,
                 population_size: int,
                 model_class: Type[nn.Module],
                 optimizer_class: Type[optim.Optimizer],
                 criterion: nn.modules.loss._Loss,
                 metric: Metric,
                 shuffle_block_classes: List[Type[nn.Module]],
                 model_params: Dict={},
                 optimizer_params: Dict={}) -> None:
        

        super().__init__(
            population_size=population_size,
            model_class=model_class,
            optimizer_class=optimizer_class,
            criterion=criterion,
            metric=metric,
            model_params=model_params,
            optimizer_params=optimizer_params
        )
        
        self._shuffle_block_classes = shuffle_block_classes

    
    def _update_population(self, 
                           cycle: int, 
                           cycles: int, 
                           fitness_scores: List[float], 
                           **kwargs: Any) -> List:

        update_history: List[Any] = [None for i in range(self._total_population_size)]
        
        with torch.no_grad():
            for i, model in enumerate(self._population):
                model_shuffle_history = {}

                for block_class in self._shuffle_block_classes:
                    shufflable_blocks: List[nn.Module] = []
                    
                    for module in model.modules():
                        if isinstance(module, block_class):
                            shufflable_blocks.append(module)
                    
                    num_blocks = len(shufflable_blocks)
                    
                    if num_blocks < 2:
                        continue
                        
                    block_states: List[Dict[str, torch.Tensor]] = []
                    for block in shufflable_blocks:
                        state = {k: v.clone() for k, v in block.state_dict().items()}
                        block_states.append(state)
                        

                    shuffled_indices = list(range(num_blocks))
                    random.shuffle(shuffled_indices)
                    
                    history_key = block_class.__name__
                    model_shuffle_history[history_key] = shuffled_indices
                    
                    for target_index, source_index in enumerate(shuffled_indices):
                        target_block = shufflable_blocks[target_index]
                        source_state = block_states[source_index]
                        
                        target_block.load_state_dict(source_state)

                update_history[i] = model_shuffle_history
                self._optimizers[i] = self._reset_optimizer(self._optimizers[i], self._population[i])

        return update_history

    
