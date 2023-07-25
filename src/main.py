# import from third parties
import torch
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.nn import Conv2d

# import from builtin (os, path, etc)


# import from own packages
from model import Model

debugging = True

model = Model(name="resnet18", 
              n_base_classes=100,
              n_novel_classes=10)

if debugging:
    out = model(torch.randn(1,3,512,512))
    for o in out:
        print(o.shape)

# first training on base_dataset: loss is ZERO on novel head

# copy the weights of the first convolution from the first conv from the base head to the novel head
with torch.no_grad(): 
    model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
    model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

# freeze the weights of everything except the novel head
for module in model.named_children():
    if module[0] != "head_novel_heatmap":
        module[1].requires_grad_(False)

# training on novel_dataset: loss on novel head is the one you used for base