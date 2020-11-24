from torch import nn
from torch import tensor, Tensor

from . import MatrixEstimator
from . import AutoEncoder
from . import Utils 


class SDAE(nn.Module):
    def __init__(self, dims, dropout=True):
        '''
            Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
            attributes. The dimensions input is the list of dimensions occurring in a single stack
            e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
            autoencoder shape [100, 10, 10, 5, 10, 10, 100].

            We use ReLUs in all encoder/decoder pairs, except for g2 of the first pair (it needs to reconstruct input 
            data that may have positive  and  negative  values, such  as  zero-mean  images) and g1 of the last pair
            (so the final data embedding retains full information).

            :param dims: list of dimensions occurring in a single stack
        '''
        super(SDAE, self).__init__()

        self.ae = []

        for idx, dim in enumerate(Utils.sliding_window_iter(dims, 2)):
            encode_relu = False if idx == (len(dims)-2) else True
            decode_relu = False if idx == 0 else True
            self.ae.append( AutoEncoder(*dim, encode_relu, decode_relu) )
          
        self.sdae = nn.Sequential(
                        MatrixEstimator(0.8), # Input matrix estimator
                        *list(map(lambda ae: ae.get_encode(dropout), self.ae)), 
                        *list(map(lambda ae: ae.get_decode(dropout), self.ae[::-1])) 
                    )

        self.ae = nn.Sequential(*self.ae)

    def layer_wise_training(self, x: Tensor) -> None:
        self.ae(x)

    def get_encode(self, dropout=False) -> nn.Sequential:
        return nn.Sequential(
            *list(map(lambda ae: ae.get_encode(dropout), self.ae))
        )


    def get_matrix(self, modules = None):
        if modules is None:
            modules = self.sdae
        
        matrix_layers = []
        for module in modules:
            if isinstance(module, (MatrixEstimator)):
                matrix_layers.append(module)
            elif isinstance(module, (nn.Sequential)):
                matrix_layers.extend( self.get_matrix(module) )
            
        return matrix_layers


    def forward(self, x: Tensor) -> Tensor:
        return self.sdae(x)