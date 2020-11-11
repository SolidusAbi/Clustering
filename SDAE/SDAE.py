from torch import nn
from torch import tensor, Tensor

class SDAE(nn.Module):
    def __init__(self, input_size):
        super(SDAE, self).__init__()

        ''' We use rectified linear units (ReLUs) in all encoder/decoder pairs, except for g2 of the first pair
            (it needs to reconstruct input data that may have positive  and  negative  values, such  as  zero-mean  images) and g1 of the last 
            pair (so the final data embedding retains full information)
        '''

        self.ae = [ AutoEncoder(input_size, 500, g1_relu=True, g2_relu=False), 
                    AutoEncoder(500, 500), 
                    AutoEncoder(500, 2000), 
                    AutoEncoder(2000, 10, g1_relu=False, g2_relu=True)
                    ]
          
        self.sdae = nn.Sequential(
                        *list(map(lambda ae: ae.get_encode(False), self.ae)), 
                        *list(map(lambda ae: ae.get_decode(False), self.ae[::-1])) 
                    )

        self.ae = nn.Sequential(*self.ae)

    def layer_wise_training(self, x: Tensor) -> None:
        self.ae(x)

    def get_encode(self, dropout=False) -> nn.Sequential:
        return nn.Sequential(
            *list(map(lambda ae: ae.get_encode(dropout), self.ae))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.sdae(x)