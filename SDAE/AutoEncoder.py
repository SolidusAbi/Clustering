from torch import nn
from torch import tensor, Tensor

''' This class contains a Layer-wise training '''
class AutoEncoder(nn.Module):
    def __init__(self, n_in, n_out, g1_relu=True, g2_relu=True):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
                nn.Dropout(p=0.2),
                # Noise(),
                nn.Linear(n_in, n_out), 
                nn.ReLU(inplace=True) if g1_relu else nn.Identity()
            )

        self.decode = nn.Sequential(
                nn.Dropout(p=0.2),
                # Noise(),
                nn.Linear(n_out, n_in),
                nn.ReLU(inplace=True) if g2_relu else nn.Identity()
            )

        for m in self.modules():
            self.weight_init(m)

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def get_encode(self, dropout=False):
        encode = []
        for module in self.encode:
            if dropout or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
                encode.append(module)
        
        return nn.Sequential(*encode)

    def get_decode(self, dropout=False):
        decode = []
        for module in self.decode:
            if dropout or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
                decode.append(module)
        
        return nn.Sequential(*decode)

    def forward(self, x: Tensor) -> Tensor:
        # Train each autoencoder individually
        x = x.detach()
        y = self.encode(x)

        if self.training:
            x_reconstruct = self.decode(y)
            loss = self.criterion(x_reconstruct, tensor(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def weight_init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decoder(x)