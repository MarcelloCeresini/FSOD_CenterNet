from typing import Dict
from torch.nn import Module, Conv2d, ReLU

class HeadRegressor(Module):

    def __init__(self, 
                 config: Dict,
                 in_channels: int) -> None:
        super().__init__()

        self.head_regressor_latent_dim = config['model']['head_regressor_latent_dim']
        self.out_channels = 4       # (size_x, size_y, offset_x, offset_y)

        self.conv1 = Conv2d(in_channels, 
                            self.head_regressor_latent_dim, 
                            kernel_size=3,
                            padding=1)
        
        self.conv2 = Conv2d(self.head_regressor_latent_dim, 
                            self.head_regressor_latent_dim, 
                            kernel_size=3,
                            padding=1)

        self.conv3 = Conv2d(self.head_regressor_latent_dim, 
                            self.head_regressor_latent_dim, 
                            kernel_size=3,
                            padding=1)

        self.conv4 = Conv2d(self.head_regressor_latent_dim, 
                            self.out_channels,
                            kernel_size=1)

        self.activation = ReLU()

        
        # TODO: do we add an activation? for the size? for the offsets? (paper doesn't mention it)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        return x