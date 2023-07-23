from torch.nn import Module
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

class Encoder(Module):
    
    def __init__(self, name: str) -> None:
        super().__init__()
        if name == "resnet18":
            model = resnet18(weights='IMAGENET1K_V1', progress = True)
            return_nodes = { "layer4.1.relu_1": "encoder" } # last layer before the classification head
            self.encoder = create_feature_extractor(model, return_nodes=return_nodes)
        else:
            raise NotImplementedError("{} is not an implemented backbone yet".format(name))
        
    def forward(self, x):
        x = self.encoder(x)
        return x