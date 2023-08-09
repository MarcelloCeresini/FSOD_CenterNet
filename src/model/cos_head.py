from torch import randn, zeros, ones, flatten, unflatten, unsqueeze, cosine_similarity
from torch.nn import Module, Parameter

from .config import Config

class CosHead(Module):
    def __init__(self,
                 n_classes: int,
                 mode: str) -> None:
        super().__init__()

        self.config = Config()

        self.range_extender = self.config.range_extender_cos_head

        self.n_classes = n_classes
        self.latent_dim = self.config.head_heatmap_latent_dim
        
        self.weights = Parameter(randn(self.n_classes, 
                                       self.latent_dim))
        
        if mode == "cos":
            self.adaptive_scale_factor = ones(self.n_classes) 
        elif mode == "adaptive":
            self.adaptive_scale_factor = Parameter(randn(self.n_classes))
        else:
            raise NotImplementedError("'{}' - this mode is not implemented yet".format(mode))

        
    def forward(self, x):
        
        out = zeros(x.shape[0], self.n_classes, x.shape[2], x.shape[3])

        # TODO: optimize, under there is a vmap example but it does not work because i instantiate
        # "zero" inside the function_for_sample_in_batch
        for b in range(x.shape[0]):
            pixel_features = flatten(x[b,:,:,:], start_dim=1)
            
            for c in range(self.n_classes):

                prototype_features_by_class = unsqueeze(self.weights[c, :], 
                                                        1)

                cos_sim = cosine_similarity(pixel_features,
                                            prototype_features_by_class,
                                            dim=0)
                
                out[b,c,:,:] = self.adaptive_scale_factor[c] * unflatten(cos_sim,
                                                                         0,
                                                                         (x.shape[2], x.shape[3])) 

        out *= self.range_extender

        return out

    # def forward(self, x):
        
    #     def function_for_sample_in_batch(x):

    #         out = zeros(self.n_classes, 
    #                     x.shape[1], 
    #                     x.shape[2])

    #         pixel_features = flatten(x, 
    #                                  start_dim=1)
            
    #         for c in range(self.n_classes):

    #             prototype_features_by_class = unsqueeze(self.weights[c, :], 
    #                                                     1)

    #             cos_sim = cosine_similarity(pixel_features,
    #                                         prototype_features_by_class,
    #                                         dim=0)
                
    #             out[c,:,:] = self.adaptive_scale_factor[c] * unflatten(cos_sim,
    #                                                                      0,
    #                                                                      (x.shape[1], x.shape[1])) 

    #     out = vmap(function_for_sample_in_batch)(x)
    #     out *= self.range_extender

    #     return out
