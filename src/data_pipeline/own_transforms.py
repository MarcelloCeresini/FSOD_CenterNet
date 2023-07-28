from typing import Any
from torchvision.transforms import RandomResizedCrop, Resize
import torchvision.transforms.functional as F
import torch as T

class RandomResizedCropOwn():
    def __init__(self, 
                 size, 
                 scale, 
                 ratio,
                 can_cut_objects: bool = True) -> None:
        
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.can_cut_objects = can_cut_objects

        if self.can_cut_objects:
            self.random_resize_crop = RandomResizedCrop(size=self.size,
                                                        scale=self.scale,
                                                        ratio=self.ratio)

    def __call__(self, sample):

        images, landmarks = sample[:3], sample[3]

        if self.can_cut_objects:
            top, left, h, w = self.random_resize_crop.get_params(images[0],
                                                                 self.scale,
                                                                 self.ratio)
        else:
            raise NotImplementedError("'Can cut objects' not implemented yet") # TODO: implement (find: top, left, h, w)

        out_images = [F.resized_crop(i,
                                     top, left, h, w, 
                                     self.size,
                                     antialias=True) 
                      for i in images]
    
        
        out_landmarks = []

        # annotations are [cpX, cpY, sizeX, sizeY] but instead crop wants [top, left, h, w]
        for l in landmarks:
            new_cp = (l["center_point"][0] - left,
                      l["center_point"][1] - top)
                        
            out_landmarks.append({
                "center_point": new_cp,
                "size":         l["size"],
                "category_id":  l["category_id"]
            })
        
        # resize
        for l in out_landmarks:
            l["center_point"] = (l["center_point"][0] * self.size[0] / w,
                                 l["center_point"][1] * self.size[1] / h)
            
            l["size"] = (l["size"][0] * self.size[0] / w,
                         l["size"][1] * self.size[1] / h)
        
        
        return out_images, out_landmarks


class RandomVerticalFlipOwn():
    def __init__(self,
                 p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 input):
        
        if T.bernoulli(T.tensor([self.p])):
            return [F.vflip(i) for i in input]
        else:
            return input


class RandomHorizontalFlipOwn():
    def __init__(self,
                 p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 input):
        
        if T.bernoulli(T.tensor([self.p])):
            return [F.hflip(i) for i in input]
        else:
            return input


class NormalizeOwn():
    def __init__(self) -> None:
        pass

    def __call__(self, image):
        return T.div(image, 255.0)
    

class ResizeAndNormalizeLabelsOwn():
    def __init__(self, 
                 size) -> None:
        self.out_size = size
        self.resize = Resize(size=self.out_size)

    def __call__(self, 
                 labels) -> T.tensor:
        _, heatmap_base, heatmap_novel = labels

        heatmap_base = self.resize(heatmap_base)
        heatmap_novel = self.resize(heatmap_novel)

        heatmap = T.cat([heatmap_base, heatmap_novel])

        heatmap = T.sum(heatmap, dim=0)
        heatmap = T.div(heatmap, T.max(heatmap))
        heatmap = T.mul(heatmap, 255.0)

        heatmap = T.stack([heatmap, T.zeros_like(heatmap), T.zeros_like(heatmap)])

        return heatmap
