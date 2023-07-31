from torchvision.transforms import ColorJitter, Compose, GaussianBlur, Resize
from torch import rand

from dataset_config import DatasetConfig
from own_transforms import RandomResizedCropOwn, ResizeOwn, RandomVerticalFlipOwn, RandomHorizontalFlipOwn, NormalizeOwn, ResizeAndNormalizeLabelsOwn
from landmarks_to_labels import LandmarksToLabels

class TransformTraining:
    def __init__(self,
                 conf: DatasetConfig = DatasetConfig()) -> None:
        
        self.random_crop = RandomResizedCropOwn(size=conf.input_to_model_resolution,
                                                scale=conf.crop_scale, # scale of the crop (before resizing) compared to original image
                                                ratio=conf.crop_ratio) # aspect ratio of the crop (before resizing) compared to original image
        
        self.random_vertical_flip = RandomVerticalFlipOwn(p=conf.p_vertical_flip)

        self.random_horizontal_flip = RandomHorizontalFlipOwn(p=conf.p_horizontal_flip)

        # TODO: check parameters and put them in config
        self.color_jitter = ColorJitter(brightness=0.2, 
                                        contrast=0.2, 
                                        saturation=0.2, 
                                        hue=0.2)
        
        sampled_sigma = rand(1) * (conf.sgb_lims[1]-conf.sgb_lims[0]) + conf.sgb_lims[0]
        
        self.gaussian_blur = GaussianBlur(kernel_size=7,
                                          sigma=sampled_sigma.item())
        
        self.normalize = NormalizeOwn()

        self.landmarks_to_labels = LandmarksToLabels(conf)


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

        image = Compose([self.color_jitter,
                         self.gaussian_blur,
                         self.normalize])(image)
        
        labels = self.landmarks_to_labels(landmarks)

        return (image, labels), landmarks, original_sample
        

class TransformTesting:
    def __init__(self,
                 conf: DatasetConfig = DatasetConfig()) -> None:
        
        self.resize = ResizeOwn(size=conf.input_to_model_resolution) # aspect ratio of the crop (before resizing) compared to original image
        
        self.normalize = NormalizeOwn()

        self.landmarks_to_labels = LandmarksToLabels(conf)


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

        return image, landmarks, original_sample


def anti_transform_testing_after_model(current_image,
                                       landmarks,
                                       original_image_size):
    '''
    Returns the landmarks in the original image size after the output of the model
    '''
    
    wi, hi = current_image.shape[1:]
    wf, hf = original_image_size

    # resize
    for l in landmarks:
        l["center_point"] = (l["center_point"][0] * wf / wi,
                             l["center_point"][1] * hf / hi)
        
        l["size"] = (l["size"][0] * wf / wi,
                     l["size"][1] * hf / hi)
        
    return current_image*255. , landmarks


        
