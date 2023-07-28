from torchvision.transforms import ColorJitter, Compose, GaussianBlur, Resize
from torch import rand

from dataset_config import DatasetConfig
from own_transforms import RandomResizedCropOwn, RandomVerticalFlipOwn, RandomHorizontalFlipOwn, NormalizeOwn, ResizeAndNormalizeLabelsOwn
from landmarks_to_labels import LandmarksToHeatmaps, GetRegressorLabels

class TransformAndAugment():
    def __init__(self,
                 conf: DatasetConfig,
                 to_be_shown: bool = False) -> None:
        
        self.to_be_shown = to_be_shown

        self.landmarks_to_labels = LandmarksToHeatmaps(conf)

        self.random_crop = RandomResizedCropOwn(size=conf.input_to_model_resolution,
                                                scale=conf.crop_scale, # scale of the crop (before resizing) compared to original image
                                                ratio=conf.crop_ratio) # aspect ratio of the crop (before resizing) compared to original image
        
        self.get_regressor_labels = GetRegressorLabels(conf)

        self.random_vertical_flip = RandomVerticalFlipOwn(p=conf.p_vertical_flip)

        self.random_horizontal_flip = RandomHorizontalFlipOwn(p=conf.p_horizontal_flip)

        self.flips = Compose([self.random_vertical_flip,
                              self.random_horizontal_flip])

        # TODO: check parameters and put them in config
        self.color_jitter = ColorJitter(brightness=0.2, 
                                        contrast=0.2, 
                                        saturation=0.2, 
                                        hue=0.2)
        
        # TODO: do we want gaussian blur?
        sampled_sigma = rand(1) * (conf.sgb_lims[1]-conf.sgb_lims[0]) + conf.sgb_lims[0]
        
        self.gaussian_blur = GaussianBlur(kernel_size=7,
                                          sigma=sampled_sigma.item())

        # if we only change the colors but not the geomerty, we don't need to change the annotations
        self.color_changes = Compose([self.color_jitter,
                                      self.gaussian_blur,
                                      # TODO: add salt and pepper noise?
                                      ])
        
        self.normalize = NormalizeOwn()
        
        self.resize_heatmaps = ResizeAndNormalizeLabelsOwn(conf.input_to_model_resolution)

    def __call__(self, sample):
        # TODO: clean up, the inputs and outputs are confused

        image, landmarks, heatmap_base, heatmap_novel = self.landmarks_to_labels(sample)

        tensors, landmarks = self.random_crop([image, heatmap_base, heatmap_novel, landmarks])

        image, heatmap_base, heatmap_novel = tensors

        regressor_label = self.get_regressor_labels(landmarks)

        sample = self.flips([image, regressor_label, heatmap_base, heatmap_novel])

        # now they are all tensors
        image, labels = sample[0], sample[1:]

        image = self.color_changes(image)

        if self.to_be_shown:
            heatmap = self.resize_heatmaps(labels)
            return {"image": image, 
                    "landmarks": landmarks,
                    "heatmap": heatmap
                    }
        else:
            return (image, labels)