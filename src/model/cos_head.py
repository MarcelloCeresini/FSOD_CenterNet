from typing import Dict

from torch import Tensor, randn, ones, cosine_similarity
from torch.nn import Module, Parameter

class CosHead(Module):
    def __init__(self,
                 config: Dict,
                 n_classes: int,
                 mode: str) -> None:

        super().__init__()
        config = config

        self.n_classes = n_classes
        self.latent_dim = config['model']['head_heatmap_latent_dim']

        self.class_prototypes = Parameter(randn(self.n_classes, self.latent_dim))

        self.tau = config['model']['range_extender_cos_head']

        if mode == "cos":
            self.adaptive_scale_factor = ones((self.n_classes, 1))
        elif mode == "adaptive":
            self.adaptive_scale_factor = Parameter(randn(self.n_classes, 1))
        else:
            raise NotImplementedError("'{}' - this mode is not implemented yet".format(mode))


    def forward(self, x: Tensor) -> Tensor:
        # We assume that input x is [batch_size, latent_dim, H, W]
        batch_size, latent_dim, H, W = x.shape

        # Expand the class prototypes so that their shape matches the input
        # Note: torch.tensor.expand returns a view of the original tensor, it doesn't allocate more memory
        cp = self.class_prototypes.T[:, :, None, None]       # [latent_dim, n_classes, 1, 1]
        cp = cp.expand(-1, -1, H, W)                         # [latent_dim, n_classes, H, W]
        cp = cp.expand(batch_size, -1, -1, -1, -1)           # [batch_size, latent_dim, n_classes, H, W]

        # Expand the input so that its shape matches the class prototypes
        h  = x[:, :, None, :, :]                             # [batch_size, latent_dim, 1, H, W]
        h  = h.expand(-1, -1, self.n_classes, -1, -1)        # [batch_size, latent_dim, n_classes, H, W]

        # Compute the cosine similarity between the class prototypes and the input
        out = cosine_similarity(cp, h, dim=1)                # [batch_size, n_classes, H, W]

        # Multiply by tau and the adaptive scale factor, then return the sigmoid of the result
        # Note: adaptive_tau must once again be expanded to match the shape of the input
        atau = self.tau * self.adaptive_scale_factor[None, :, :, None].expand(batch_size, -1, H, W).to(out.device)
        out *= atau
        return out.sigmoid()
