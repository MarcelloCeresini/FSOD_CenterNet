from torch.nn import Module, Conv2d, ReLU, Sigmoid

class HeadHeatmap(Module):

    def __init__(self, 
                 in_channels: int, 
                 n_classes: int) -> None:
        super().__init__()

        self.out_channels = n_classes

        self.conv1 = Conv2d(in_channels, 
                            in_channels, 
                            kernel_size=3)
        
        self.activation1 = ReLU()

        self.conv2 = Conv2d(in_channels, 
                            n_classes, 
                            kernel_size=1)
        
        self.activation2 = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        return x