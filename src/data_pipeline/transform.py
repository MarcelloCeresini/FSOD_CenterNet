from typing import Dict, List, Tuple
import torch as T
from torchvision.transforms import ColorJitter, Compose, GaussianBlur

from .own_transforms import RandomResizedCropOwn, ResizeOwn, \
    RandomVerticalFlipOwn, RandomHorizontalFlipOwn, NormalizeOwn, \
    ResizeAndNormalizeLabelsOwn
from .landmarks_to_labels import LandmarksToLabels, LandmarksTransform


class TransformTraining:
    def __init__(self,
                 config: Dict,
                 base_classes: List = [],
                 novel_classes: List = []) -> None:
        
        self.max_detections = config['data']['max_detections']

        self.random_crop = RandomResizedCropOwn(
            size=config['data']['input_to_model_resolution'],
            scale=config['data']['augmentations']['crop_scale'], # scale of the crop (before resizing) compared to original image
            ratio=config['data']['augmentations']['crop_ratio']) # aspect ratio of the crop (before resizing) compared to original image
        
        self.random_vertical_flip = RandomVerticalFlipOwn(
            p=config['data']['augmentations']['p_vertical_flip'])

        self.random_horizontal_flip = RandomHorizontalFlipOwn(
            p=config['data']['augmentations']['p_horizontal_flip'])

        self.color_jitter = ColorJitter(
            brightness=config['data']['augmentations']['brightness_jitter'], 
            contrast=config['data']['augmentations']['contrast_jitter'], 
            saturation=config['data']['augmentations']['saturation_jitter'], 
            hue=config['data']['augmentations']['hue_jitter'])
        
        sampled_sigma = T.rand(1) * \
            (config['data']['augmentations']['sgb_lims'][1]-config['data']['augmentations']['sgb_lims'][0]) \
            + config['data']['augmentations']['sgb_lims'][0]
        
        self.gaussian_blur = GaussianBlur(kernel_size=7,
                                          sigma=sampled_sigma.item())
        
        self.normalize = NormalizeOwn()

        self.landmarks_to_labels = LandmarksToLabels(config, base_classes, novel_classes)
        self.transform_landmarks = LandmarksTransform(config, base_classes, novel_classes)
 

    def __call__(self, 
                 original_sample):
        '''
        Returns:
            (image, labels): tuple of tensors to be given as input to the training of the model
            landmarks: useful for visualization
            original_sample: useful for visualization
        '''
        transformed_sample = Compose([self.random_crop,
                                      self.random_vertical_flip,
                                      self.random_horizontal_flip])(original_sample)

        image, landmarks = transformed_sample["image"], transformed_sample["landmarks"]

        n_landmarks = len(landmarks)

        image = Compose([self.color_jitter,
                         self.gaussian_blur,
                         self.normalize])(image)
        
        labels = self.landmarks_to_labels(landmarks)

        padded_landmarks = self.transform_landmarks(landmarks)

        return image, labels, n_landmarks, padded_landmarks


class TransformTesting:
    def __init__(self,
                 config: Dict,
                 base_classes: List = [],
                 novel_classes: List = []) -> None:
        
        self.resize = ResizeOwn(size=config['data']['input_to_model_resolution']) 
        
        self.normalize = NormalizeOwn()

        self.landmarks_to_labels = LandmarksToLabels(config, base_classes, novel_classes)
        self.transform_landmarks = LandmarksTransform(config, base_classes, novel_classes)


    def __call__(self, 
                 original_sample):
        '''
        Returns:
            image: tensor to be given as input to the model
            landmarks: useful for visualization and evaluation (but they are resized)
            original_sample: useful for visualization and evaluation (they contain the original landmarks)
        '''
        transformed_sample = self.resize(original_sample)

        image, landmarks = transformed_sample["image"], transformed_sample["landmarks"]

        image = self.normalize(image)

        n_landmarks = len(landmarks)

        padded_landmarks = self.transform_landmarks(landmarks)

        return image, 0, n_landmarks, padded_landmarks


# def anti_transform_testing_after_model(current_image,
#                                        landmarks,
#                                        original_image_size) -> Tuple[T.tensor]:
#     '''
#     Returns the landmarks in the original image size after the output of the model
#     '''
    
#     wi, hi = current_image.shape[1:]
#     wf, hf = original_image_size

#     # resize
#     for l in landmarks:
#         l["center_point"] = (l["center_point"][0] * wf / wi,
#                              l["center_point"][1] * hf / hi)
        
#         l["size"] = (l["size"][0] * wf / wi,
#                      l["size"][1] * hf / hi)
        
#     return current_image*255. , landmarks


        
