# import from third parties


# import from builtin (os, path, etc)


# import from own packages
import torch
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.nn import Conv2d

from model import Model

model = Model("resnet18")

model(torch.randn(1,3,512,512)).shape