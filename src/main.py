# import from third parties
import torch


# import from builtin (os, path, etc)
import sys

# import from own packages
from model import Model
from data_pipeline import DatasetFromCocoAnnotations, TransformTraining, TransformTesting
from evaluation import Evaluate

debugging = True


model = Model(encoder_name="resnet18", 
              n_base_classes=100,
              n_novel_classes=10,
              head_base_heatmap_mode="CosHead",
              head_novel_heatmap_mode="AdaptiveCosHead")

# base dataset
dataset_base_train = DatasetFromCocoAnnotations(annotations_path="base_train.json", # TODO: change this path
                                                images_dir="always_the_same_dir",
                                                transform=TransformTraining())

dataset_novel_train = DatasetFromCocoAnnotations(annotations_path="novel_train.json", # TODO: change this path
                                                 images_dir="always_the_same_dir",
                                                 transform=TransformTraining())

dataset_base_test = DatasetFromCocoAnnotations(annotations_path="base_test.json", # TODO: change this path
                                               images_dir="always_the_same_dir",
                                               transform=TransformTesting())

dataset_novel_test = DatasetFromCocoAnnotations(annotations_path="novel_test.json", # TODO: change this path
                                                images_dir="always_the_same_dir",
                                                transform=TransformTesting())

dataset_full_test = DatasetFromCocoAnnotations(annotations_path="full_test.json", # TODO: change this path
                                               images_dir="always_the_same_dir",
                                               transform=TransformTesting())

if debugging:
    print("Dataset base train length: ", len(dataset_base_train))
    sample, landmarks, original_image_size = dataset_base_train[0]
    # use "show_images.py" functions to show the sample / samples


# first training on base_dataset: loss is ZERO on novel head
for sample, _, _ in dataset_base_train:
    input_image, labels = sample
    # forward pass
    pass

# copy the weights of the first convolution from the first conv from the base head to the novel head
with torch.no_grad(): 
    model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
    model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

# freeze the weights of everything except the novel head
for module in model.named_children():
    if module[0] != "head_novel_heatmap":
        module[1].requires_grad_(False)

# training on novel_dataset: loss on novel head is the one you used for base
for sample, _, _ in dataset_novel_train:
    # forward pass
    input_image, labels = sample
    pass


# evaluation on base_dataset
metrics_base = Evaluate(model,
                        dataset_base_test)

# evaluation on novel_dataset
metrics_novel = Evaluate(model,
                         dataset_novel_test)

# aggregation and print results
metrics_full = Evaluate(model,
                        dataset_full_test)