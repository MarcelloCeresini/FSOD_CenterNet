from torch.nn import Module
from torchvision.ops import DeformConv2d
from torch.nn import ConvTranspose2d, Conv2d

class Neck(Module):

    def __init__(self, init_channels: int) -> None:
        super().__init__()
        out_channels_block_1 = int(init_channels/2)
        self.offset_conv_1 = Conv2d(init_channels, 2*init_channels, kernel_size=3, padding=1)
        self.def_conv1 = DeformConv2d(
            in_channels=init_channels, 
            out_channels=out_channels_block_1, 
            kernel_size=3,
            padding=1
        )
        self.trans_conv1 = ConvTranspose2d(out_channels_block_1, out_channels_block_1, kernel_size=3, stride=2, padding=1, output_padding=1) # to check

    def forward(self, x):
        print("feature map shape", x)
        y = self.offset_conv_1(x)
        print("offsets shape", y)
        x = self.def_conv1(x, y)
        print("after deformable conv", x.shape)
        x = self.trans_conv1(x)
        print("after transposed conv", x.shape)
        return x
    


        