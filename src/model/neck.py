from torch.nn import Module
from torchvision.ops import DeformConv2d
from torch.nn import ConvTranspose2d, Conv2d

from .deformable_conv_2d import DeformableConv2d


class Upsample(Module):
    def __init__(self,
                 in_channels) -> None:
        
        super().__init__()

        self.out_channels = int(in_channels/2)

        self.def_conv1 = DeformableConv2d(in_channels, 
                                          self.out_channels)
        
        self.trans_conv1 = ConvTranspose2d(self.out_channels, 
                                           self.out_channels, 
                                           kernel_size=3, 
                                           stride=2, 
                                           padding=1, 
                                           output_padding=1)

    def forward(self, x):
        x = self.def_conv1(x)
        x = self.trans_conv1(x)
        return x
    

class Neck(Module):
    def __init__(self,
                 in_channels: int) -> None:
        
        super().__init__()

        self.upsample1 = Upsample(in_channels)
        self.upsample2 = Upsample(int(in_channels/2))
        self.upsample3 = Upsample(int(in_channels/4))

        self.out_channels = self.upsample3.out_channels

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        return x
    


        