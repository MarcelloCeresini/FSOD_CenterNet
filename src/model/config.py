from typing import Any


class Config:
    def __init__(self) -> None:

        self.head_regressor_latent_dim = 64 # output dimension of the first 3x3 convolution in the regressor head
        self.head_heatmap_latent_dim = 64 # output dimension of the first 3x3 convolution in the heatmap head
        
        self.range_extender_cos_head = 10.0
