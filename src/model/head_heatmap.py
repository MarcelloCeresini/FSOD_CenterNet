from typing import Dict
from torch.nn import Module, Conv2d, ReLU, Sigmoid, Softmax

from .cos_head import CosHead


class HeadHeatmap(Module):

    def __init__(self, 
                 config: Dict,
                 in_channels: int, 
                 n_classes: int, 
                 mode: str) -> None:
        super().__init__()

        self.less_convs = config['model']['less_convs']
        self.softmax_activation = config["model"]["softmax_activation"]
        self.out_channels = n_classes
        self.conv1 = Conv2d(in_channels, 
                            in_channels, 
                            kernel_size=3,
                            padding=1)

        if not self.less_convs:
            self.conv2 = Conv2d(in_channels, 
                                in_channels, 
                                kernel_size=3,
                                padding=1)
            self.conv3 = Conv2d(in_channels, 
                                in_channels, 
                                kernel_size=3,
                                padding=1)

        self.activation1 = ReLU()
        
        if mode == "convolution":
            self.second_block = Conv2d(in_channels, 
                                n_classes, 
                                kernel_size=1)
        elif mode == "CosHead":
            self.second_block = CosHead(config, n_classes, mode="cos")
        elif mode == "AdaptiveCosHead":
            self.second_block = CosHead(config, n_classes, mode="adaptive")
        else:
            raise NotImplementedError("'{}' - this mode is not implemented yet".format(mode))
        
        self.activation2 = Sigmoid()

        if self.softmax_activation:
            self.pixel_wise_rescaling = Conv2d(in_channels=n_classes,
                                               out_channels=1,
                                               kernel_size=1)
            self.activation3 = Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        if not self.less_convs:
            x = self.activation1(x)
            x = self.conv2(x)
            x = self.activation1(x)
            x = self.conv3(x)
        x = self.activation1(x)
        x = self.second_block(x)
        if not self.softmax_activation:
            x = self.activation2(x)
        else:
            y = self.pixel_wise_rescaling(x)
            y = self.activation2(x)
            x = self.activation3(x)
            x = x*y
        return x