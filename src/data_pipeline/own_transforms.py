from torchvision.transforms import RandomResizedCrop
import torchvision.transforms.functional as F
import torch

class RandomResizedCropOwn():
    def __init__(self, 
                 size, 
                 scale, 
                 ratio) -> None:
        
        self.size = size
        self.scale = scale
        self.ratio = ratio

        self.random_resize_crop = RandomResizedCrop(size=self.size,
                                                    scale=self.scale,
                                                    ratio=self.ratio)

    def __call__(self, sample):

        image, landmarks = sample["image"], sample["landmarks"]

        top, left, h, w = self.random_resize_crop.get_params(image, 
                                                        self.scale, 
                                                        self.ratio)
        
        image = F.resized_crop(image,
                               top, left, h, w, 
                               self.size,
                               antialias=True)
        
        accepted = []

        # annotations are [cpX, cpY, sizeX, sizeY] but instead crop wants [top, left, h, w]
        for l in landmarks:
            new_cp = (l["center_point"][0] - left,
                      l["center_point"][1] - top)
                        
            # TODO: is it ok if we accept bboxes ONLY IF THEIR CENTER IS IN THE CROP? 
            # TODO: do we need some stats to understand how many we reject?
            if (0 <= new_cp[0] <= w) and (0 <= new_cp[1] <= h):

                accepted.append({
                    "center_point": new_cp,
                    "size":         l["size"],
                    "category_id":  l["category_id"]
                })

        # resize
        for l in accepted:
            l["center_point"] = (l["center_point"][0] * self.size[0] / w,
                                 l["center_point"][1] * self.size[1] / h)
            
            l["size"] = (l["size"][0] * self.size[0] / w,
                         l["size"][1] * self.size[1] / h)
            
        return {"image": image, 
                "landmarks": accepted}


class RandomVerticalFlipOwn():
    def __init__(self,
                 p: float = 0.5) -> None:
        self.p = p

    def __call__(self, 
                 sample):
        image, landmarks = sample["image"], sample["landmarks"]
        _, h = F.get_image_size(image)

        if torch.bernoulli(torch.tensor([self.p])):
            image = F.vflip(image)

            for l in landmarks:
                l["center_point"] = (l["center_point"][0],
                                     h - l["center_point"][1])
        
        return {"image": image, 
                "landmarks": landmarks}


class RandomHorizontalFlipOwn():
    def __init__(self,
                 p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        
        w, _ = F.get_image_size(image)

        if torch.bernoulli(torch.tensor([self.p])):

            image = F.hflip(image)

            for l in landmarks:
                l["center_point"] = (w - l["center_point"][0],
                                     l["center_point"][1])
        
        return {"image": image, 
                "landmarks": landmarks}


class NormalizeOwn():
    def __init__(self) -> None:
        pass
    
    def __call__(self, image):
        return torch.div(image, 255.0)