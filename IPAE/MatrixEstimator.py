import math 
import torch
from functools import reduce
from torch import Tensor
from torch import nn

from IPDL import TensorKernel

class MatrixEstimator(nn.Module):
    def __init__(self, gamma: float, normalize_dim = True):
        super(MatrixEstimator, self).__init__()
        
        self.gamma = gamma
        self.normalize_dim = normalize_dim
        self.A = torch.rand((10,10))        

    def set_gamma(self, gamma: float) -> None:
        self.gamma = gamma

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x

        n = x.size(0)
        d = x.size(1) if len(x.shape) == 2 else reduce(lambda x, y: x*y, x.shape[1:])
        
        sigma = self.gamma * (n ** (-1 / (4 + d)))
        if self.normalize_dim:
            sigma = sigma * math.sqrt(d)

        self.A = TensorKernel.RBF(x.detach().flatten(1), sigma).cpu() / n

        return x

    def __repr__(self):
        return "MatrixEstimator(gamma={})".format(self.gamma)

