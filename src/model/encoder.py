from torch.nn import Module
from torchvision.models.resnet import resnet18, resnet34, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

class Encoder(Module):
    
    def __init__(self, name: str) -> None:
        super().__init__()

        if name == "resnet18":
            model = resnet18(weights='IMAGENET1K_V1', progress = True)
            return_nodes = { "layer4.1.bn2": "encoder" }
        elif name == 'resnet34':
            model = resnet34(weights='IMAGENET1K_V1', progress = True)
            return_nodes = { 'layer4.2.bn2': 'encoder' }
        elif name == 'resnet50':
            model = resnet50(weights='IMAGENET1K_V2', progress = True)
            return_nodes = { 'layer4.2.relu': 'encoder' }
        else:
            raise NotImplementedError("'{}' is not an implemented backbone yet".format(name))

        self.encoder = create_feature_extractor(model, return_nodes=return_nodes)
        
    def forward(self, x):
        x = self.encoder(x)
        return x