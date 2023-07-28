import torch as T
import numpy as np

from dataset_config import DatasetConfig

class LandmarksToHeatmaps:
    def __init__(self,
                 conf: DatasetConfig) -> None:
        
        # model outputs (out_reg, out_heat_base, out_heat_novel)
        self.output_resolution = conf.output_resolution
        self.regressor_label_size = [4, *self.output_resolution]

        self.output_stride = conf.output_stride
        self.min_IoU_for_gaussian_radius = conf.min_IoU_for_gaussian_radius
        self.n_base_classes = conf.n_base_classes
        self.n_novel_classes = conf.n_novel_classes

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
        
        masked_heatmap  = heatmap[cy - space_top: cy + space_bottom,
                                  cx - space_left: cx + space_right]
        masked_gaussian = gaussian[radius - space_top: radius + space_bottom,
                                   radius - space_left: radius + space_right]
        
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            T.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
        return heatmap


    def __call__(self, 
                 sample) -> tuple():
        
        image, landmarks = sample["image"], sample["landmarks"]

        img_size = image.shape[1:3]
        
        heatmap_base = T.zeros([self.n_base_classes, *img_size])
        heatmap_novel = T.zeros([self.n_novel_classes, *img_size])

        for l in landmarks:
            cp_idx = [int(np.floor(l["center_point"][0])),
                      int(np.floor(l["center_point"][1]))]

            if (cat := l["category_id"]) < (n_bc := self.n_base_classes):
                heatmap_base[cat, ...] = self.draw_gaussian(heatmap_base[cat, ...],
                                                            [*cp_idx, *l["size"]])
            else:
                heatmap_novel[cat, ...] = self.draw_gaussian(heatmap_novel[cat-n_bc, ...],
                                                             [*cp_idx, *l["size"]])
        return image, landmarks, heatmap_base, heatmap_novel
    

class GetRegressorLabels:
    '''
    Create the offset and size labels for the regressor head ONLY AFTER the image has been cropped and resized.
    '''
    def __init__(self,
                 conf: DatasetConfig) -> None:
        
        self.output_resolution = conf.output_resolution
        self.regressor_label_size = [4, *self.output_resolution]
        self.output_stride = conf.output_stride
        
    def __call__(self,
                 landmarks):
        
        # TODO: it could be a sparse tensor
        regressor_label = T.zeros(self.regressor_label_size) # already in low res

        for l in landmarks:
            low_res_cp = [l["center_point"][0] / self.output_stride[0],
                          l["center_point"][1] / self.output_stride[1]]
            
            size_small = [l["size"][0] / self.output_stride[0],
                          l["size"][1] / self.output_stride[1]]
            
            lr_cp_idx = [int(np.floor(low_res_cp[0])),
                         int(np.floor(low_res_cp[1]))]
            
            if (0 <= lr_cp_idx[0] <= self.output_resolution[0]) and \
               (0 <= lr_cp_idx[1] <= self.output_resolution[1]):
            
                offset = [low_res_cp[0] - lr_cp_idx[0],
                          low_res_cp[1] - lr_cp_idx[1]]

                regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = offset[0]
                regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = offset[1]
                regressor_label[2, lr_cp_idx[0], lr_cp_idx[1]] = size_small[0]
                regressor_label[3, lr_cp_idx[0], lr_cp_idx[1]] = size_small[1]

        return regressor_label