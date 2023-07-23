from torchvision.models.resnet import ResNet, resnet18

class Backbone:
    def __init__(self, name: str) -> ResNet:
        if name == "resnet18":
            return resnet18(weights='IMAGENET1K_V1', progress = True) 
        else:
            raise NotImplementedError("{} is not an implemented backbone yet".format(name))