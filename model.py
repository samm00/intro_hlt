import torch as th
import numpy as np
from transformers import FlaubertModel
import torch.nn as nn

class Model(th.nn.Module):
    def __init__(self, transformer):
        super(Cascade, self).__init__()
        if transformer == 'flaubert':
            self.transformer = FlaubertModel()

    def forward(self,tweet):
        # TODO 