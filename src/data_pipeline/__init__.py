import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

import json, os

from .transform import TransformTraining, TransformTesting


class DatasetFromCocoAnnotations(Dataset):

    def __init__(self, 
                 annotations_path: str, 
                 images_dir: str, 
                 transform = None,
                 to_be_shown: bool = False) -> None:
        super().__init__()

        with open(annotations_path, "r") as f:
            self.annotation_file = json.load(f)
        
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_file["images"])

    def __getitem__(self, idx):
        '''
        Returns a sample of the dataset. If no transform is passed, the sample is a dictionary with:
            - image
            - landmarks:
                - category_id
                - center_point
                - size
            - original_image_size
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        current_image_id = self.annotation_file["images"][idx]["id"]

        img_name = os.path.join(self.images_dir,
                                self.annotation_file["images"][idx]["file_name"])
        
        image = read_image(img_name)

        annotation_for_image = []
        for a in self.annotation_file["annotations"]:
            if a["image_id"] == current_image_id:
                # a["bbox"] is top [left x position, top left y position, width, height]
                center_point = (a["bbox"][0] + a["bbox"][2]/2, 
                                a["bbox"][1] + a["bbox"][3]/2)
                
                size = (a["bbox"][2], 
                        a["bbox"][3])
                
                annotation_for_image.append({
                    "category_id": a["category_id"],
                    "center_point": center_point,
                    "size": size
                })


        sample = {'image': image, 
                  "landmarks": annotation_for_image}

        if isinstance(self.transform, (TransformTraining, TransformTesting)):
            
            sample, transformed_landmarks, original_sample = self.transform(sample)

            return sample, transformed_landmarks, original_sample

        else:
            return sample
        

def get_data_loaders(annotations_paths: list(str), 
                     images_dir: str, 
                     transform = None,
                     batch_size: int = None,
                     num_workers: int = 0,
                     pin_memory: bool = False,
                     drop_last: bool = False,
                     shuffle: bool = False) -> list(DataLoader):
    
    return [DataLoader(dataset=DatasetFromCocoAnnotations(annotations_path=annotations_path,
                                                         images_dir=images_dir,
                                                         transform=transform),
                       batch_size=batch_size,
                       num_workers=num_workers,
                       pin_memory=pin_memory,
                       drop_last=drop_last,
                       shuffle=shuffle) 
                for annotations_path in annotations_paths]
