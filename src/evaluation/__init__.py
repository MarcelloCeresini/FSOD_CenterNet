import torch as T
import torch.nn.functional as NNF
import torchvision.transforms.functional as TF
import numpy as np
from torchmetrics.detection import MeanAveragePrecision

from data_pipeline.transform import anti_transform_testing_after_model

class Evaluate:
    '''
    Takes as input a model and a dataset WITHOUT TRANSFORMATIONS and evaluates the model
    '''

    def __init__(self,
                 model, 
                 dataset):
        
        self.model = model
        self.dataset = dataset
        self.stride = self.dataset.transform.conf.output_stride

        self.metric = MeanAveragePrecision(box_format="cxcywh")

        # self.IoU_threshold_sets = conf.IoU_threshold_sets

        # self.confusion_matrices = {key: [np.zeros(2,2) for _ in self.IoU_threshold_sets[key]] for key in self.IoU_threshold_sets}

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

        for image, landmarks, original_sample in self.dataset:
            # both image and landmarks will be resized to model_input_size
            pred = self.model(image)

            complete_heatmaps = T.cat(pred[1], 
                                  pred[2])

            idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)

            landmarks_pred = self.landmarks_from_idxs(pred[0],
                                                      complete_heatmaps,
                                                      idxs_tensor)
            
            landmarks_pred_resized_to_original = anti_transform_testing_after_model(image,
                                                                                    landmarks_pred,
                                                                                    TF.get_image_size(original_sample["image"]))

            pred_to_metric = []
            for l in landmarks_pred_resized_to_original:
                pred_to_metric.append({
                    "boxes": l["center_point"] + l["size"],
                    "labels": l["category_id"],
                    "scores": l["confidence_score"]
                })

            gt_to_metric = []
            for l in original_sample["landmarks"]:
                gt_to_metric.append({
                    "boxes": l["center_point"] + l["size"],
                    "labels": l["category_id"]
                })

            self.metric.update(preds=pred_to_metric, 
                               target=gt_to_metric)
            
        return self.metric.compute()
        

