from torch.nn import Module, Conv2d, ReLU

class HeadRegressor(Module):

    def __init__(self, 
                 in_channels: int) -> None:
        super().__init__()

        self.out_channels = 4

        self.conv1 = Conv2d(in_channels, 
                            in_channels, 
                            kernel_size=3)
        
        self.activation = ReLU()

        self.conv2 = Conv2d(in_channels, 
                            4, # (size_x, size_y, offset_x, offset_y)
                            kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x