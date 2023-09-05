import math
from typing import List

import numpy as np
import torch as T

from .dataset_config import DatasetConfig


class LandmarksToLabels:
    def __init__(self,
                 conf: DatasetConfig,
                 base_class_list: List,
                 novel_class_list: List) -> None:
        
        self.base_classes           = base_class_list
        self.novel_classes          = novel_class_list

        # model outputs (out_reg, out_heat_base, out_heat_novel)
        self.output_resolution      = conf.output_resolution
        self.regressor_label_size   = [4, *self.output_resolution]
        self.heatmap_base_size      = [len(self.base_classes), *self.output_resolution]
        self.heatmap_novel_size     = [len(self.novel_classes), *self.output_resolution]

        self.output_stride          = conf.output_stride
        self.min_IoU_for_gaussian_radius = conf.min_IoU_for_gaussian_radius

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

        x, y = np.ogrid[-radius:radius+1,
                        -radius:radius+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return T.Tensor(h)
    

    def draw_gaussian(self, heatmap, landmark):

        cx, cy, sx, sy = landmark
        cx, cy = int(cx), int(cy)

        radius = int(self.gaussian_radius((sx, sy), 
                                          min_overlap=self.min_IoU_for_gaussian_radius))
        
        gaussian = self.gaussian2D_kernel(radius, 
                                          sigma=radius / 3)

        height, width = heatmap.shape[0:2]
        
        space_left, space_right = min(cx, radius), min(width - cx, radius + 1)
        space_top, space_bottom = min(cy, radius), min(height -cy, radius + 1)

        # masked_heatmap  = heatmap[cx - space_left: cx + space_right,
        #                           cy - space_top: cy + space_bottom]
        # masked_gaussian = gaussian[radius - space_left: radius + space_right,
        #                            radius - space_top: radius + space_bottom]
        
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

        for l in landmarks:
            low_res_cp = [l["center_point"][i] / self.output_stride[i] 
                          for i in range(len(l["center_point"]))]

            lr_cp_idx = [math.floor(low_res_cp[i]) 
                         for i in range(len(low_res_cp))]

            # TODO: it was int, but I think it should be float?? Otherwise it's always 0
            offset = [low_res_cp[i] - lr_cp_idx[i]
                      for i in range(len(low_res_cp))]

            # TODO: is "size" this large? shouldn't it be normalized?
            # TODO: also: does the regress label only contain the offset in the exact
            # center point? Shouldn't it contain offsets within the gaussian radius too?
            regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][0]
            regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][1]
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