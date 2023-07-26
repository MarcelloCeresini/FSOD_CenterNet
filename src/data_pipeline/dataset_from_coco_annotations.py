import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import json, os

class DatasetFromCocoAnnotations(Dataset):

    def __init__(self, 
                 annotations_path: str, 
                 images_dir: str, 
                 transform=None) -> None:
        super().__init__()

        with open(annotations_path, "r") as f:
            self.annotation_file = json.load(f)
        
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_file["images"])

    def __getitem__(self, idx):
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

        if self.transform:
            sample = self.transform(sample)

        return sample