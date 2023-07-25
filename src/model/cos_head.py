from torch.nn import Module, Parameter, Sigmoid
from torch import randn, zeros, ones, cosine_similarity

from .config import Config

class CosHead(Module):
    def __init__(self,
                 n_classes: int,
                 mode: str) -> None:
        super().__init__()

        self.range_extender = Config().range_extender_cos_head

        self.n_classes = n_classes
        self.latent_dim = Config().head_heatmap_latent_dim
        
        self.weights = Parameter(randn(self.n_classes, 
                                       self.latent_dim))
        
        if mode is None:
            self.adaptive_scale_factor = ones(self.n_classes) 
        elif mode == "adaptive":
            self.adaptive_scale_factor = Parameter(randn(self.n_classes))
        else:
            raise NotImplementedError("'{}' - this mode is not implemented yet".format(mode))

        
    def forward(self, x):

        # TODO: optimize this
        out = zeros(self.n_classes, x.shape[1], x.shape[2])
        for c in range(self.n_classes):
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    out[c,i,j] = cosine_similarity(x[:,i,j], self.weights[c,:]) * self.adaptive_scale_factor[c]
        
        out *= self.range_extender

        return out

                
