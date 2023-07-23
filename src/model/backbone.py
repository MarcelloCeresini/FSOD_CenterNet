import torchvision.models.resnet as Resnets

class Backbone:
    def __init__(self, name: str) -> None:
        if name == "resnet18":
            return Resnets.resnet18(weights='IMAGENET1K_V1', progress = True) 
        else:
            raise NotImplementedError("{} is not an implemented backbone yet".format(name))