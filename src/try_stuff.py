import torch

a = torch.randn(3,4,4)



for o in out:
    print(o.shape)
    print(o)

    def forward(self, x):
        
        def function_for_sample_in_batch(x):

            out = zeros(self.n_classes, 
                        x.shape[1], 
                        x.shape[2])

            pixel_features = flatten(x, 
                                     start_dim=1)
            
            for c in range(self.n_classes):

                prototype_features_by_class = unsqueeze(self.weights[c, :], 
                                                        1)

                cos_sim = cosine_similarity(pixel_features,
                                            prototype_features_by_class,
                                            dim=0)
                
                out[c,:,:] = self.adaptive_scale_factor[c] * unflatten(cos_sim,
                                                                         0,
                                                                         (x.shape[1], x.shape[1])) 

        out = vmap(function_for_sample_in_batch)(x)
        out *= self.range_extender

        return out