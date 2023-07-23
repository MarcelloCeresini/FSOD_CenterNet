from torch.nn import Module
from torchvision.ops import DeformConv2d
from torch.nn import ConvTranspose2d, Conv2d

from .deformable_conv_2d import DeformableConv2d

class Neck(Module):

    def __init__(self, init_channels: int) -> None:
        super().__init__()
        out_channels_block_1 = int(init_channels/2)
        self.def_conv1 = DeformableConv2d(init_channels, out_channels_block_1)
        self.trans_conv1 = ConvTranspose2d(out_channels_block_1, out_channels_block_1, kernel_size=3, stride=2, padding=1, output_padding=1)

        out_channels_block_2 = int(init_channels/4)
        self.def_conv2 = DeformableConv2d(out_channels_block_1, out_channels_block_2)
        self.trans_conv2 = ConvTranspose2d(out_channels_block_2, out_channels_block_2, kernel_size=3, stride=2, padding=1, output_padding=1)

        out_channels_block_3 = int(init_channels/8)
        self.def_conv3 = DeformableConv2d(out_channels_block_2, out_channels_block_3)
        self.trans_conv3 = ConvTranspose2d(out_channels_block_3, out_channels_block_3, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # print("feature map shape", x.shape)
        # y = self.offset_conv_1(x)
        # print("offsets shape", y.shape)
        # x = self.def_conv1(x, y)
        # print("after deformable conv", x.shape)
        # x = self.trans_conv1(x)
        # print("after transposed conv", x.shape)
        x = self.def_conv1(x)
        x = self.trans_conv1(x)
        x = self.def_conv2(x)
        x = self.trans_conv2(x)
        x = self.def_conv3(x)
        x = self.trans_conv3(x)
        return x
    


        