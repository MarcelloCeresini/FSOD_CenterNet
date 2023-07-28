from torchvision.transforms import ColorJitter, Compose, GaussianBlur, Resize
from torch import rand

from dataset_config import DatasetConfig
from own_transforms import RandomResizedCropOwn, RandomVerticalFlipOwn, RandomHorizontalFlipOwn, NormalizeOwn, ResizeAndNormalizeLabelsOwn
from landmarks_to_labels import LandmarksToLabels

class TransformAndAugment():
    def __init__(self,
                 conf: DatasetConfig,
                 to_be_shown: bool = False) -> None:
        
        self.to_be_shown = to_be_shown

        self.random_crop = RandomResizedCropOwn(size=conf.input_to_model_resolution,
                                                scale=conf.crop_scale, # scale of the crop (before resizing) compared to original image
                                                ratio=conf.crop_ratio) # aspect ratio of the crop (before resizing) compared to original image
        
        self.random_vertical_flip = RandomVerticalFlipOwn(p=conf.p_vertical_flip)

        self.random_horizontal_flip = RandomHorizontalFlipOwn(p=conf.p_horizontal_flip)

        self.composed = Compose([self.random_crop,
                                 self.random_vertical_flip,
                                 self.random_horizontal_flip,
                                 ])

        # TODO: check parameters and put them in config
        self.color_jitter = ColorJitter(brightness=0.2, 
                                        contrast=0.2, 
                                        saturation=0.2, 
                                        hue=0.2)
        
        sampled_sigma = rand(1) * (conf.sgb_lims[1]-conf.sgb_lims[0]) + conf.sgb_lims[0]
        
        self.gaussian_blur = GaussianBlur(kernel_size=7,
                                          sigma=sampled_sigma.item())

        # if we only change the colors but not the geomerty, we don't need to change the annotations
        list_of_compositions_no_landmarks = [self.color_jitter,
                                             self.gaussian_blur,
                                            # TODO: add salt and pepper noise?
                                            ]
        
        self.normalize = NormalizeOwn()

        list_of_compositions_only_landmarks = [LandmarksToLabels(conf)]

        if self.to_be_shown:
            list_of_compositions_only_landmarks.append(ResizeAndNormalizeLabelsOwn(conf.input_to_model_resolution))
        else:
            list_of_compositions_no_landmarks.append(self.normalize)
        
        self.composed_no_landmarks = Compose(list_of_compositions_no_landmarks)
        self.compose_only_landmarks = Compose(list_of_compositions_only_landmarks)

    def __call__(self, sample):
        sample = self.composed(sample)

        image, landmarks = sample["image"], sample["landmarks"]

        image = self.composed_no_landmarks(image)
        labels = self.compose_only_landmarks(landmarks)
        # labels = self.transform_landmarks_into_labels(landmarks)

        if self.to_be_shown:
            return {"image": image, 
                    "landmarks": landmarks,
                    "labels": labels
                    }
        else:
            return (image, labels)