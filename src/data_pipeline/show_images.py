import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import torch as T
from PIL.Image import blend

from dataset_from_coco_annotations import DatasetFromCocoAnnotations
from transform import TransformAndAugment
from dataset_config import DatasetConfig

dataset = DatasetFromCocoAnnotations(annotations_path="/Users/marcelloceresini/github/FSOD_CenterNet/data/fsod/annotations/fsod_train_short.json",
                                     images_dir="/Users/marcelloceresini/github/FSOD_CenterNet/data/fsod/images",
                                     transform=TransformAndAugment(DatasetConfig(), 
                                                                   to_be_shown=True))

img_list = []
annotation_list = []

for i in dataset:
    img_list.append(i["image"])
    annotation_list.append(i["landmarks"])

plt.rcParams["savefig.bbox"] = 'tight'

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def rewrite_bbox(img_size, annotation):
    list_of_bboxes = []
    if not isinstance(annotation, list):
        annotation = [annotation]

    for a in annotation:
        cp = a["center_point"]
        size = a["size"]
        xyxy_bbox = [max(0, cp[0] - size[0]/2), 
                     max(0, cp[1] - size[1]/2), 
                     min(img_size[0], cp[0] + size[0]/2), 
                     min(img_size[1], cp[1] + size[1]/2)]
        list_of_bboxes.append(xyxy_bbox)
        # list_of_bboxes.append(box_convert(torch.tensor([[cp[0], cp[1], size[0], size[1]]]), "cxcywh", "xyxy"))
        assert 0 <= xyxy_bbox[0] < xyxy_bbox[2] <= img_size[0], f"bbox: {xyxy_bbox}, img_size: {img_size}, cp: {cp}, size: {size}"
        assert 0 <= xyxy_bbox[1] < xyxy_bbox[3] <= img_size[1], f"bbox: {xyxy_bbox}, img_size: {img_size}, cp: {cp}, size: {size}"
    
    if len(list_of_bboxes) == 0:
        return T.tensor([])
    
    out = T.tensor(list_of_bboxes)

    return out

### SHOW BOUDING BOXES
# grid = [draw_bounding_boxes(i["image"], 
#                             rewrite_bbox(F.get_image_size(i["image"]), i["landmarks"]),
#                             width=4) for i in dataset]
# show(grid)

### SHOW HEATMAPS
grid = [F.pil_to_tensor(blend(F.to_pil_image(i["image"]), 
                              F.to_pil_image(i["heatmap"]), 
                              alpha=0.4))
        for i in dataset]
show(grid)

plt.show()
