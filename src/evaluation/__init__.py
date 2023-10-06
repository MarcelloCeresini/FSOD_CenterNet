import torch as T
import torch.nn.functional as NNF
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from time import time

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
                 dataset,
                 device,
                 conf,
                 prefix=""):
        
        self.model   = model
        self.dataset = dataset
        self.prefix  = prefix
        self.device  = device
        self.conf = conf
    
        self.metric = MeanAveragePrecision(box_format="cxcywh")

    def get_heatmap_maxima_idxs(self, 
                                complete_heatmaps):
        
        pooled_heatmaps = NNF.max_pool2d(complete_heatmaps,
                                       3,
                                       stride=1,
                                       padding=1)

        return (complete_heatmaps == pooled_heatmaps)

    def landmarks_from_idxs(self,
                            regressor_pred: T.tensor,
                            complete_heatmaps: T.tensor,
                            idxs_tensor_mask: T.tensor):
        # TODO: import a conf file here (maybe after uniting every config)

        n_classes, output_width, output_height = idxs_tensor_mask.shape

        max_detections = 100
        
        landmarks_pred = {
                    "boxes": T.zeros(max_detections,4),
                    "labels": T.zeros(max_detections).to(T.int32),
                    "scores": T.zeros(max_detections)
                }

        # landmarks_pred = [{"category_id": 0,
        #                    "center_point": [0, 0],
        #                    "size": [0, 0],
        #                    "confidence_score": 0
        #                    }] * max_detections

        # flattens it so we can use topk
        confidence_scores = T.masked_select(complete_heatmaps,
                                            idxs_tensor_mask)

        # the i-th element in top_k_scores has the i-th highest confidence score in the image, but its index refers to its position in "confidence_scores"
        # that has only n values, where n is the number of times that idxs_tensor_mask is true (it doesn't have n_classes*output_width*output_height indices)
        top_k_scores = T.topk(confidence_scores,
                              max_detections)
        
        # this retrieves all of the (flattened) indices (of the output image) where the classification has a peak
        flattened_idxs = T.nonzero(T.flatten(idxs_tensor_mask)).reshape(-1)

        # this retrieves only the top "max_detections" of them (but still, flattened)
        flattened_top_k_idxs = flattened_idxs[top_k_scores.indices]

        src = T.ones(n_classes*output_width*output_height).to(device="cuda")
        flattened_top_k_mask = T.zeros_like(src).to(device="cuda").scatter_add_(dim=0,          
                                                                                index=flattened_top_k_idxs,
                                                                                src=src).bool()

        unflattened_top_k_mask = T.unflatten(flattened_top_k_mask,
                                             dim=0,
                                             sizes=(n_classes, output_width, output_height))
        
        unflattened_top_k_idxs = T.nonzero(unflattened_top_k_mask)

        regressor_pred_repeated = regressor_pred.repeat(n_classes,1,1).reshape(n_classes, *regressor_pred.shape)

        size_x = T.masked_select(regressor_pred_repeated[:,0,:,:],
                                 unflattened_top_k_mask)
        size_y = T.masked_select(regressor_pred_repeated[:,1,:,:],
                                 unflattened_top_k_mask)
        off_x = T.masked_select(regressor_pred_repeated[:,2,:,:],
                                 unflattened_top_k_mask)
        off_y = T.masked_select(regressor_pred_repeated[:,3,:,:],
                                 unflattened_top_k_mask)

        category = unflattened_top_k_idxs[:,0]
        center_idx_x = unflattened_top_k_idxs[:,1]
        center_idx_y = unflattened_top_k_idxs[:,2]

        center_coord_x = center_idx_x+off_x
        center_coord_y = center_idx_y+off_y

        for i, (c, cx, cy, sx, sy, score) in \
            enumerate(zip(category, center_coord_x, center_coord_y, size_x, size_y, confidence_scores)):

                landmarks_pred["boxes"][i,0] = cx
                landmarks_pred["boxes"][i,1] = cy
                landmarks_pred["boxes"][i,2] = sx
                landmarks_pred["boxes"][i,3] = sy
                landmarks_pred["labels"][i] = c
                landmarks_pred["scores"][i] = score
                
                # landmarks_pred[i]["category_id"] = c
                # landmarks_pred[i]["center_point"][0] = cx
                # landmarks_pred[i]["center_point"][1] = cy
                # landmarks_pred[i]["size"][0] = sx
                # landmarks_pred[i]["size"][1] = sy
                # landmarks_pred[i]["confidence_score"] = score

        return landmarks_pred


    def __call__(self):

        for counter, (image_batch, _, n_landmarks_batch, padded_landmarks) in tqdm(enumerate(self.dataset), 
                                                                             total=len(self.dataset)):
            # both image and landmarks will be resized to model_input_size
            reg_pred_batch, heat_base_pred_batch, heat_novel_pred_batch = \
                self.model(image_batch.to(self.device))

            if heat_novel_pred_batch is None:
                heat_novel_pred_batch = [None] * heat_base_pred_batch.shape[0]

            for i, (reg_pred, heat_base_pred, heat_novel_pred, n_landmarks) in \
                enumerate(zip(reg_pred_batch, heat_base_pred_batch, heat_novel_pred_batch, n_landmarks_batch)):

                if heat_novel_pred is None:
                    complete_heatmaps = heat_base_pred
                else:
                    complete_heatmaps = T.cat(heat_base_pred, heat_novel_pred)

                idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)

                # TODO: SORT BY CONFIDENCE SCORES AND CUT TO MAX NUMBER OF PREDICTIONS
                landmarks_pred = self.landmarks_from_idxs(
                    reg_pred,
                    complete_heatmaps,
                    idxs_tensor
                )

                # Recreate the landmarks from the padded / batched version
                landmarks_gt = []
                for l in range(n_landmarks.item()):
                    landmarks_gt.append({
                        "center_point":(padded_landmarks[l]["center_point"][0][i].item(), 
                                        padded_landmarks[l]["center_point"][1][i].item()),
                        "size":(padded_landmarks[l]["center_point"][0][i].item(), 
                                padded_landmarks[l]["center_point"][1][i].item()),
                        "category_id":padded_landmarks[l]["category_id"][i].item()}
                    )


                # pred_to_metric = {
                #     "boxes": [],
                #     "labels": [],
                #     "scores": []
                # }
                # for l in landmarks_pred:
                #     pred_to_metric["boxes"].append(l["center_point"] + l["size"])
                #     pred_to_metric["labels"].append(l["category_id"])
                #     pred_to_metric["scores"].append(l["confidence_score"])

                # pred_to_metric["boxes"] = T.tensor(pred_to_metric["boxes"])
                # pred_to_metric["labels"] = T.tensor(pred_to_metric["labels"])
                # pred_to_metric["scores"] = T.tensor(pred_to_metric["scores"])

                gt_to_metric = {
                    "boxes": [],
                    "labels": []
                }
                for l in landmarks_gt:
                    gt_to_metric["boxes"].append(l["center_point"] + l["size"])
                    gt_to_metric["labels"].append(l["category_id"])

                gt_to_metric["boxes"] = T.tensor(gt_to_metric["boxes"])
                gt_to_metric["labels"] = T.tensor(gt_to_metric["labels"])

                # TODO: THE METRIC ONLY WORKS IF PRED AND GT HAVE THE SAME NUMBER OF ELEMENTS
                self.metric.update(
                    preds=[landmarks_pred], 
                    target=[gt_to_metric]
                )
            
            if counter > 10:
                break
            
        result = self.metric.compute()
        result2 = {}

        if self.prefix != "":
            for key in result:
                result2[self.prefix + key] = result[key]

        return result2
        

