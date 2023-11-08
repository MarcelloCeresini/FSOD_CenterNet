import math
from typing import List, Dict

import numpy as np
import torch as T

class LandmarksToLabels:
    def __init__(self,
                 config: Dict,
                 base_class_list: List,
                 novel_class_list: List) -> None:
        
        self.base_classes           = base_class_list
        self.novel_classes          = novel_class_list

        # model outputs (out_reg, out_heat_base, out_heat_novel)
        self.input_resolution       = config['data']['input_to_model_resolution']
        self.output_stride          = config['data']['output_stride']

        self.output_resolution      = [x // self.output_stride[i]
            for i, x in enumerate(self.input_resolution)]
        
        self.regressor_label_size   = [4, *self.output_resolution]
        self.heatmap_base_size      = [len(self.base_classes), *self.output_resolution]
        self.heatmap_novel_size     = [len(self.novel_classes), *self.output_resolution]
        self.min_IoU_for_gaussian_radius = config['data']['min_IoU_for_gaussian_radius']
        self.max_detections = max(config['data']['max_detections'])

    def gaussian_radius(self, det_size, min_overlap):
        '''
        Credit: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
        '''
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return min(r1, r2, r3)
        
        
    def gaussian2D_kernel(self, radius, sigma=1):
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return T.Tensor(h)
    

    def draw_gaussian(self, heatmap, landmark):

        cx, cy, sx, sy = landmark
        cx, cy = int(cx), int(cy)

        # TODO: quick fix: force to at least 1
        radius = max(int(self.gaussian_radius((sx, sy), min_overlap=self.min_IoU_for_gaussian_radius)), 1)
        
        gaussian = self.gaussian2D_kernel(radius, 
                                          sigma=radius / 3)

        height, width = heatmap.shape[0:2]
        
        space_left, space_right = min(cx, radius), min(width - cx, radius + 1)
        space_top, space_bottom = min(cy, radius), min(height -cy, radius + 1)
        
        masked_heatmap  = heatmap[cy - space_top: cy + space_bottom,
                                  cx - space_left: cx + space_right]
        masked_gaussian = gaussian[radius - space_top: radius + space_bottom,
                                   radius - space_left: radius + space_right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            T.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        return heatmap


    def __call__(self, 
                 landmarks) -> tuple():

        regressor_label = T.zeros(self.regressor_label_size)
        heatmap_base    = T.zeros(self.heatmap_base_size)
        heatmap_novel   = T.zeros(self.heatmap_novel_size)

        for i, l in enumerate(landmarks):

            if i >= self.max_detections:
                break

            low_res_cp = [l["center_point"][i] / self.output_stride[i] 
                          for i in range(len(l["center_point"]))]

            lr_cp_idx = [min(math.floor(low_res_cp[i]), self.output_resolution[i] - 1)
                         for i in range(len(low_res_cp))]

            offset = [low_res_cp[i] - lr_cp_idx[i]
                      for i in range(len(low_res_cp))]

            regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][0] / \
                (self.input_resolution[0] if self.config['training']['normalize_size'] else 1)
            regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][1] / \
                (self.input_resolution[1] if self.config['training']['normalize_size'] else 1)
            regressor_label[2, lr_cp_idx[0], lr_cp_idx[1]] = offset[0]
            regressor_label[3, lr_cp_idx[0], lr_cp_idx[1]] = offset[1]

            cat = l["category_id"]
            if cat in self.novel_classes:
                cat_idx = self.novel_classes.index(cat)
                heatmap_novel[cat_idx, ...] = self.draw_gaussian(heatmap_novel[cat_idx, ...], 
                                                                 [*lr_cp_idx, *l["size"]])
            else:
                cat_idx = self.base_classes.index(cat)
                heatmap_base[cat_idx, ...] = self.draw_gaussian(heatmap_base[cat_idx, ...], 
                                                                [*lr_cp_idx, *l["size"]])

        return regressor_label, heatmap_base, heatmap_novel
    

class LandmarksTransform:
    def __init__(self, 
                 config: Dict,
                 base_class_list: List,
                 novel_class_list: List):

        self.config = config
        self.max_detections = max(config['data']['max_detections'])
        self.base_class_list = base_class_list
        self.novel_class_list = novel_class_list
        self.output_stride = config["data"]["output_stride"]
        self.input_resolution = config['data']['input_to_model_resolution']


    def __call__(self,
             landmarks):
        
        padded_landmarks = {
            "boxes": T.zeros(self.max_detections,4),
            "labels": T.zeros(self.max_detections).to(T.int32),
        }

        for i, l in enumerate(landmarks):
            padded_landmarks["boxes"][i,0] = l["center_point"][0] / self.output_stride[0]
            padded_landmarks["boxes"][i,1] = l["center_point"][1] / self.output_stride[1]
            if self.config['training']['normalize_size']:
                padded_landmarks["boxes"][i,2] = l["size"][0] / self.input_resolution[0]
                padded_landmarks["boxes"][i,3] = l["size"][1] / self.input_resolution[1]
            padded_landmarks["labels"][i]  = self.base_class_list.index(l["category_id"])   \
                if l['category_id'] in self.base_class_list \
                else self.novel_class_list.index(l["category_id"]) + len(self.base_class_list)

        return padded_landmarks