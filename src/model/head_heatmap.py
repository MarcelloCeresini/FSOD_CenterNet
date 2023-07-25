from torch.nn import Module, Conv2d, ReLU, Sigmoid

from .cos_head import CosHead


class HeadHeatmap(Module):

    def __init__(self, 
                 in_channels: int, 
                 n_classes: int, 
                 mode: str) -> None:
        super().__init__()

        self.out_channels = n_classes

        self.conv1 = Conv2d(in_channels, 
                            in_channels, 
                            kernel_size=3)
        
        self.activation1 = ReLU()

        if mode is None:

            self.second_block = Conv2d(in_channels, 
                                n_classes, 
                                kernel_size=1)
            
        elif mode == "CosHead":

            self.second_block = CosHead(n_classes)

        elif mode == "AdaptiveCosHead":

            self.second_block = CosHead(n_classes, mode="adaptive")

        else:
            raise NotImplementedError("'{}' - this mode is not implemented yet".format(mode))

        self.activation2 = Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.second_block(x)
        x = self.activation2(x)
        return x