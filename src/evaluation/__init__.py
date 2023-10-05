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
                            regressor_pred,
                            complete_heatmaps,
                            idxs_tensor_mask):
        
        n_classes, output_width, output_height = idxs_tensor_mask.shape

        max_detections = 100
        
        landmarks_pred = []

        confidence_scores = T.masked_select(complete_heatmaps,
                                            idxs_tensor_mask)
        
        # the i-th element in top_k_scores has the i-th highest confidence score in the image, but its index refers to its position in "confidence_scores"
        # that has only n values, where n is the number of times that idxs_tensor_mask is true (it doesn't have n_classes*output_width*output_height indices)

        top_k_scores = T.topk(confidence_scores,
                              max_detections)
        
        

        full_idxs = T.nonzero(idxs_tensor_mask, as_tuple=False)



        # this retrieves all of the (flattened) indices (of the output image) where the classification has a peak
        flattened_idxs = T.nonzero(T.flatten(idxs_tensor_mask))

        # this retrieves only the top "max_detections" of them (but still, flattened)
        flattened_top_k_idxs = flattened_idxs[top_k_scores.indices]

        unflattened_top_k_idxs = T.unflatten(flattened_top_k_idxs,
                                             dim=0,
                                             sizes=(-1, output_width, output_height))

        center_data = T.zeros(max_detections, 100)


        # for c in range(idxs_tensor_mask.shape[0]):
        
        #     # TODO: check dimensions of idxs_tensor and select only topk given their confidence scores, 
        #     # then use them to mask_select center_data

        #     # TODO: import a conf file here (maybe after uniting every config)
        #     top_k_confidence = T.topk(confidence_scores,
        #                               100)

        #     center_data = T.masked_select(regressor_pred, 
        #                                   idxs_tensor[c]).reshape((-1, 4))

        #     center_coords_x, center_coords_y = T.where(idxs_tensor[c])

        #     for info, score, cp_idx_x, cp_idx_y in \
        #         zip(center_data, confidence_scores, center_coords_x, center_coords_y):

        #         size_x, size_y, off_x, off_y = info                
        #         cx, cy = (cp_idx_x+off_x), (cp_idx_y+off_y)

        #         landmarks_pred.append({
        #             "category_id": c,       # TODO: CHECK THAT THE FIRST CLASS IS 0 IN THE ORIGINAL DATASET, ...
        #             "center_point": [cx.item(), cy.item()],
        #             "size": [size_x.item(), size_y.item()],
        #             "confidence_score": score.item()
        #         })
        
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

                tic = time()
                if heat_novel_pred is None:
                    complete_heatmaps = heat_base_pred
                else:
                    complete_heatmaps = T.cat(heat_base_pred, heat_novel_pred)

                idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)
                print("get_heatmap_maxima_idxs", time()-tic)

                tic = time()
                # TODO: SORT BY CONFIDENCE SCORES AND CUT TO MAX NUMBER OF PREDICTIONS
                landmarks_pred = self.landmarks_from_idxs(
                    reg_pred,
                    complete_heatmaps,
                    idxs_tensor
                )
                print("landmarks_from_idxs", time()-tic)
                tic = time()
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

                print("retrieval", time()-tic)

                tic = time()
                pred_to_metric = {
                    "boxes": [],
                    "labels": [],
                    "scores": []
                }
                for l in landmarks_pred:
                    pred_to_metric["boxes"].append(l["center_point"] + l["size"])
                    pred_to_metric["labels"].append(l["category_id"])
                    pred_to_metric["scores"].append(l["confidence_score"])

                pred_to_metric["boxes"] = T.tensor(pred_to_metric["boxes"])
                pred_to_metric["labels"] = T.tensor(pred_to_metric["labels"])
                pred_to_metric["scores"] = T.tensor(pred_to_metric["scores"])

                pred_to_metric = [pred_to_metric]

                gt_to_metric = {
                    "boxes": [],
                    "labels": []
                }
                for l in landmarks_gt:
                    gt_to_metric["boxes"].append(l["center_point"] + l["size"])
                    gt_to_metric["labels"].append(l["category_id"])

                gt_to_metric["boxes"] = T.tensor(gt_to_metric["boxes"])
                gt_to_metric["labels"] = T.tensor(gt_to_metric["labels"])

                gt_to_metric = [gt_to_metric]

                print("conversion", time()-tic)
                tic = time()
                # TODO: THE METRIC ONLY WORKS IF PRED AND GT HAVE THE SAME NUMBER OF ELEMENTS
                self.metric.update(
                    preds=pred_to_metric, 
                    target=gt_to_metric
                )
                print("update ",time()-tic)
            
            if counter > 10:
                break
            
        result = self.metric.compute()
        
        if self.prefix != "":
            for key in result:
                result[self.prefix + key] = result.pop(key)

        return result
        

