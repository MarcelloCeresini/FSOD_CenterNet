from torch.nn import Module, Conv2d

from .encoder import Encoder
from .neck import Neck

class Model(Module):

    def __init__(self, name) -> None:
        super().__init__()

        self.encoder = Encoder(name)

        # get the number of channels of the last conv_layer of the encoder
        encoded_channels = list(filter(
            lambda x: type(x)==Conv2d, 
            list(self.encoder.modules()))
        )[-1].out_channels

        self.neck = Neck(encoded_channels)

    def forward(self, x):
        x = self.encoder(x)["encoder"] # needed because the encoder comes from "create_feature_extractor"
        x = self.neck(x)
        return x