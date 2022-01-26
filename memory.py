from collections import namedtuple, deque

import random
import numpy as np
import torch

Transition = namedtuple('Transition', ('obs', 'action', 'next_obs', 'rew', 'done'))
    
class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):    
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        instances = [self.memory[idx] for idx in indexes]
        batch = Transition(*zip(*instances))
        return batch
    
    def __len__(self):
        return len(self.memory)