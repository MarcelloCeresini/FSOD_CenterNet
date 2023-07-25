from torch.nn import Module, Conv2d

from .encoder import Encoder
from .neck import Neck
from .head_regressor import HeadRegressor
from .head_heatmap import HeadHeatmap

class Model(Module):

    def __init__(self, name, n_base_classes, n_novel_classes) -> None:
        super().__init__()

        self.encoder = Encoder(name)

        # get the number of channels of the last conv_layer of the encoder
        encoded_channels = list(filter(
            lambda x: type(x)==Conv2d, 
            list(self.encoder.modules()))
        )[-1].out_channels

        self.neck = Neck(encoded_channels)

        head_input_channels = self.neck.out_channels

        self.head_regressor = HeadRegressor(head_input_channels) # 2**3 because of the upsampling in the neck (halve the channels three times)

        self.head_base_heatmap = HeadHeatmap(head_input_channels,
                                             n_base_classes)
        
        self.head_novel_heatmap = HeadHeatmap(head_input_channels,
                                              n_novel_classes)
        
    def forward(self, x):
        x = self.encoder(x)["encoder"] # needed because the encoder comes from "create_feature_extractor"
        x = self.neck(x)

        out_reg = self.head_regressor(x) # (batch_size, 4, init_size_x/4, init_size_y/4)
        out_heat_base = self.head_base_heatmap(x) # (batch_size, n_base_classes, init_size_x/4, init_size_y/4)
        out_heat_novel = self.head_novel_heatmap(x) # (batch_size, n_novel_classes, init_size_x/4, init_size_y/4)

        return (out_reg, out_heat_base, out_heat_novel)