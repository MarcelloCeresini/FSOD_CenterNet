import torch as T
import torch.nn.functional as F
import numpy as np

from .eval_config import EvalConfig
from data_pipeline import TransformAndAugment

class Evaluate:
    '''
    Takes as input a model and a dataset WITHOUT TRANSFORMATIONS and evaluates the model
    '''

    def __init__(self,
                 model, 
                 dataset,
                 conf: EvalConfig = EvalConfig()):
        
        self.model = model
        self.dataset = dataset
        self.conf = conf
        self.stride = self.dataset.transform.conf.output_stride

        self.dataset.set_transform(TransformAndAugment(testing=True))

        self.IoU_threshold_sets = conf.IoU_threshold_sets

        self.confusion_matrices = {key: [np.zeros(2,2) for _ in self.IoU_threshold_sets[key]] for key in self.IoU_threshold_sets}

    def get_heatmap_maxima_idxs(self, 
                                complete_heatmaps):
        
        pooled_heatmaps = F.max_pool2d(complete_heatmaps,
                                       3,
                                       stride=1,
                                       padding=1)

        return (complete_heatmaps == pooled_heatmaps)
    
    def landmarks_from_idxs(self,
                            regressor_pred,
                            complete_heatmaps,
                            idxs_tensor):
        
        landmarks_pred = []

        for c in range(idxs_tensor.shape[0]):
            center_data = T.masked_select(regressor_pred, 
                                          idxs_tensor[c])
            confidence_scores = T.masked_select(complete_heatmaps[c],
                                                idxs_tensor[c])
            
            for info, score, cp_idx in zip(center_data, confidence_scores, idxs_tensor[c]):
                size_x, size_y, off_x, off_y = info
                cx, cy = (cp_idx[0]+off_x) * self.stride[0] , (cp_idx[1]+off_y) * self.stride[1]

                landmarks_pred.append({
                    "category_id": c,
                    "center_point": [cx, cy],
                    "size": [size_x, size_y],
                    "consfidence_score": score
                })


    def __call__(self):

        for image, landmarks in self.dataset:
            # both image and landmarks will be resized to model_input_size
            pred = self.model(image)

            complete_heatmaps = T.cat(pred[1], 
                                  pred[2])

            idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)

            landmarks_pred = self.landmarks_from_idxs(pred[0],
                                                      complete_heatmaps,
                                                      idxs_tensor)

            for th_set in self.IoU_threshold_sets:
                for th in th_set:
                    for l_gt, l_pred in zip(landmarks, landmarks_pred):
                        pass

        
        # Calculate the precision at every recall value(0 to 1 with a step size of 0.01), 
        # then it is repeated for IoU thresholds of 0.55,0.60,…,.95.

