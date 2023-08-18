import torch as T
import torch.nn.functional as NNF
import torchvision.transforms.functional as TF
from torchmetrics.detection import MeanAveragePrecision

from data_pipeline.transform import anti_transform_testing_after_model

class Evaluate:
    '''
    Takes as input a model and a dataset WITHOUT TRANSFORMATIONS and evaluates the model
        
    Returns:
    
    - ``map_dict``: A dictionary containing the following key-values:
        - map: (:class:`~torch.Tensor`), global mean average precision
        - map_small: (:class:`~torch.Tensor`), mean average precision for small objects
        - map_medium:(:class:`~torch.Tensor`), mean average precision for medium objects
        - map_large: (:class:`~torch.Tensor`), mean average precision for large objects
        - mar_1: (:class:`~torch.Tensor`), mean average recall for 1 detection per image
        - mar_10: (:class:`~torch.Tensor`), mean average recall for 10 detections per image
        - mar_100: (:class:`~torch.Tensor`), mean average recall for 100 detections per image
        - mar_small: (:class:`~torch.Tensor`), mean average recall for small objects
        - mar_medium: (:class:`~torch.Tensor`), mean average recall for medium objects
        - mar_large: (:class:`~torch.Tensor`), mean average recall for large objects
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds), mean average precision at
            IoU=0.50
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds), mean average precision at
            IoU=0.75
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average precision per
            observed class
        - mar_100_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average recall for 100
            detections per image per observed class
        - classes (:class:`~torch.Tensor`), list of all observed classes
    '''
    def __init__(self,
                 model, 
                 dataset):
        
        self.model   = model
        self.dataset = dataset
        # TODO: not clear what you're trying to get here but dataloader has no attibute transform
        self.stride  = self.dataset.transform.conf.output_stride

        self.metric = MeanAveragePrecision(box_format="cxcywh")

    def get_heatmap_maxima_idxs(self, 
                                complete_heatmaps):
        
        pooled_heatmaps = NNF.max_pool2d(complete_heatmaps,
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
                cp_idx_x, cp_idx_y = cp_idx

                cx, cy = (cp_idx_x+off_x) * self.stride[0] , \
                            (cp_idx_y+off_y) * self.stride[1]

                landmarks_pred.append({
                    "category_id": c,
                    "center_point": [cx, cy],
                    "size": [size_x, size_y],
                    "confidence_score": score
                })


    def __call__(self):

        for image_for_model, _, original_sample in self.dataset:
            # both image and landmarks will be resized to model_input_size
            pred = self.model(image_for_model)

            complete_heatmaps = T.cat(pred[1], 
                                      pred[2])

            idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)

            landmarks_pred = self.landmarks_from_idxs(pred[0],
                                                      complete_heatmaps,
                                                      idxs_tensor)
            
            _, landmarks_pred_resized_to_original = anti_transform_testing_after_model(image_for_model,
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
        

