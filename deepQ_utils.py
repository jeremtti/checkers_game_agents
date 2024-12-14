from board import Board
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image, ImageDraw
import itertools
import pandas as pd
import random
from typing import List, Tuple, Deque, Optional, Callable



class QNetwork(torch.nn.Module):

    def __init__(self, n_input: int, nn_l1: int=100, nn_l2: int=100):
        super(QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_input, nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x
    

def initialize_q_network(board_side: int) -> QNetwork:
    """
    Function that initializes the QNetwork.
    """
    size = int(board_side**2/2) # The size of the board
    return QNetwork(7*size)


class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer: torch.optim.Optimizer, lr_decay: float, last_epoch: int = -1, min_lr: float = 1e-6):
        
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch=-1)

    def get_lr(self):
        
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]


class EpsilonScheduler:

    def __init__(self,
                 epsilon_start: float,
                 epsilon_min: float,
                 epsilon_decay:float,
                 ):
        
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def random_action(self):
        real = np.random.rand()
        if real < self.epsilon:
            answer  = True
            self.decay_epsilon()
        else:
            answer  = False
        return answer

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)