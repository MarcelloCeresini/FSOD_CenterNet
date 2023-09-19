from typing import Any


class Config:
    def __init__(self) -> None:

        self.head_regressor_latent_dim = 64 # output dimension of the first 3x3 convolution in the regressor head
        self.head_heatmap_latent_dim = 64 # output dimension of the first 3x3 convolution in the heatmap head
        
        self.range_extender_cos_head = 10.0 # Fixed to 10. by the paper

        self.alpha_loss = 2 # TODO: should be hyperparameter of focal loss
        self.beta_loss = 4 # TODO: should be hyperparameter of focal loss
        self.lambda_size_loss = 0.1 # TODO: is it ok?
        self.lambda_offset_loss = 1. # TODO: is it ok?

        