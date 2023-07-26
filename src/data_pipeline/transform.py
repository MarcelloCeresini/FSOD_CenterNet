from torchvision.transforms import ColorJitter, Compose, Normalize

from dataset_config import DatasetConfig
from own_transforms import RandomResizedCropOwn, RandomVerticalFlipOwn, RandomHorizontalFlipOwn, NormalizeOwn

class TransformAndAugment():
    def __init__(self,
                 conf: DatasetConfig) -> None:
        
        self.random_crop = RandomResizedCropOwn(size=conf.input_to_model_resolution,
                                                scale=conf.crop_scale, # scale of the crop (before resizing) compared to original image
                                                ratio=conf.crop_ratio) # aspect ratio of the crop (before resizing) compared to original image
        
        self.random_vertical_flip = RandomVerticalFlipOwn(p=conf.p_vertical_flip)

        self.random_horizontal_flip = RandomHorizontalFlipOwn(p=conf.p_horizontal_flip)

        self.composed = Compose([self.random_crop,
                                 self.random_vertical_flip,
                                 self.random_horizontal_flip,
                                 # TODO: add transformation of annotations to tensor
                                 ])

        # TODO: check parameters
        self.color_jitter = ColorJitter(brightness=0.2, 
                                        contrast=0.2, 
                                        saturation=0.2, 
                                        hue=0.2)
        
        self.normalize = NormalizeOwn()

        # if we only change the colors but not the geomerty, we don't need to change the annotations
        self.composed_no_landmarks = Compose([self.color_jitter,
                                              # TODO: add smoothing / salt and pepper noise?
                                              self.normalize,
                                              ])

    def __call__(self, sample):
        sample = self.composed(sample)

        image, labels = sample["image"], sample["landmarks"]

        image = self.composed_no_landmarks(image)

        # TODO: remove this and leave the one below once you have finished
        return {"image": image, 
                "landmarks": labels}
    
        return (image, labels)