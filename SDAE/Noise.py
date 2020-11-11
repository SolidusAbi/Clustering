from torch import nn
from torch import tensor

''' Pre-imputation '''
class Noise(nn.Module):
    def __init__(self):
        super(Noise, self).__init__()
    
    def forward(self, x):
        if self.training:
            return x * (tensor(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        
        return x