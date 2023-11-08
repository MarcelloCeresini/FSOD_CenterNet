import torch as T
import torch.nn.functional as NNF
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from torchvision.ops import box_iou, box_convert

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
                 config,
                 prefix=""):
        
        self.model   = model
        self.dataset = dataset
        self.prefix  = prefix
        self.device  = device
        self.config  = config
        self.metric = MeanAveragePrecision(box_format="cxcywh", class_metrics=True, 
                                           max_detection_thresholds=self.config['data']['max_detections'])
        self.metric_only_regression = MeanAveragePrecision(box_format="cxcywh")

    @T.no_grad()
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
        n_classes, output_width, output_height = idxs_tensor_mask.shape

        num_detections = T.sum(idxs_tensor_mask).to('cpu')
        num_detections = min(max(self.config['data']['max_detections']), num_detections)
        
        landmarks_pred = {
            "boxes": T.zeros(num_detections,4),
            "labels": T.zeros(num_detections).to(T.int32),
            "scores": T.zeros(num_detections)
        }

        # Flattens it so we can use topk
        confidence_scores = T.masked_select(complete_heatmaps, idxs_tensor_mask)

        # The i-th element in top_k_scores has the i-th highest confidence score in the image, 
        # but its index refers to its position in "confidence_scores" (which is a flattened tensor
        # tthat has as many elements as idxs_tensor_mask's true values, or peaks).
        # Instead, we will need a n_classes*output_width*output_height tensor to get indices
        top_k_scores = T.topk(confidence_scores, num_detections)
        
        # This retrieves all of the (flattened) indices (of the output image) where the classification has a peak
        flattened_idxs = T.nonzero(T.flatten(idxs_tensor_mask)).reshape(-1)

        # this retrieves only the top "num_detections" of them (but still, flattened)
        flattened_top_k_idxs = flattened_idxs[top_k_scores.indices]

        base_mask = T.zeros(n_classes*output_width*output_height).to(device=self.device)
        # Populates the mask with 1s for topk indices
        base_mask[flattened_top_k_idxs] += 1
        mask = base_mask.to(dtype=T.bool)

        top_k_mask = T.unflatten(mask, dim=0, sizes=(n_classes, output_width, output_height))
        top_k_idxs = T.nonzero(top_k_mask)

        regressor_pred_repeated = regressor_pred.repeat(n_classes,1,1,1)

        size_x = T.masked_select(regressor_pred_repeated[:,0,:,:],
                                 top_k_mask)
        size_y = T.masked_select(regressor_pred_repeated[:,1,:,:],
                                 top_k_mask)
        off_x = T.masked_select(regressor_pred_repeated[:,2,:,:],
                                 top_k_mask)
        off_y = T.masked_select(regressor_pred_repeated[:,3,:,:],
                                 top_k_mask)

        category = top_k_idxs[:,0]
        center_idx_y = top_k_idxs[:,1]
        center_idx_x = top_k_idxs[:,2]

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

        return landmarks_pred

    def resize_landmarks(self, landmarks):
        landmarks["boxes"][:,0] *= self.config['data']['output_stride'][0]
        landmarks["boxes"][:,1] *= self.config['data']['output_stride'][1]
        if self.config['training']['normalize_size']:
            landmarks["boxes"][:,2] *= self.config['data']['input_to_model_resolution'][0]
            landmarks["boxes"][:,3] *= self.config['data']['input_to_model_resolution'][1]
        return landmarks

    @T.no_grad()
    def __call__(self, is_novel=False, is_full:bool=False):
        assert not (is_novel and is_full), "is_novel and is_full cannot be both True"

        only_classification_metric = 0
        n=0
        thrs = self.config["eval"]["threshold_only_classification_metric"]

        for counter, (image_batch, _, n_landmarks_batch, padded_landmarks) in tqdm(
                enumerate(self.dataset), total=len(self.dataset), position=1 + int(is_novel), 
                leave=False, desc="Evaluation " + self.prefix):

            # both image and landmarks will be resized to model_input_size
            reg_pred_batch, heat_base_pred_batch, heat_novel_pred_batch = \
                self.model(image_batch.to(self.device))

            pred_batch = []
            gt_batch   = []
            pred_positive_batch = []
            gt_positive_batch = []

            if heat_novel_pred_batch is None:
                complete_heatmaps_batch = heat_base_pred_batch
            elif is_full:
                complete_heatmaps_batch = T.cat([heat_base_pred_batch, heat_novel_pred_batch], dim=1)
            elif is_novel:
                complete_heatmaps_batch = T.cat([T.zeros_like(heat_base_pred_batch), heat_novel_pred_batch], dim=1)
            else:
                raise NotImplementedError

            # iteration on batch_size
            for i, (reg_pred, complete_heatmaps, n_landmarks) in \
                enumerate(zip(reg_pred_batch, complete_heatmaps_batch, n_landmarks_batch)):


                idxs_tensor = self.get_heatmap_maxima_idxs(complete_heatmaps)

                landmarks_pred = self.landmarks_from_idxs(
                    reg_pred,
                    complete_heatmaps,
                    idxs_tensor
                )

                landmarks_gt = {
                    "boxes": padded_landmarks["boxes"][i,:n_landmarks,:],
                    "labels": padded_landmarks["labels"][i,:n_landmarks]
                }

                landmarks_pred = self.resize_landmarks(landmarks_pred)
                landmarks_gt = self.resize_landmarks(landmarks_gt)

                pred_batch.append(landmarks_pred)
                gt_batch.append(landmarks_gt)

                landmarks_pred_positive = landmarks_pred.copy()
                landmarks_gt_positive = landmarks_gt.copy()

                landmarks_pred_positive["labels"] = T.zeros_like(landmarks_pred_positive["labels"])
                landmarks_gt_positive["labels"] = T.zeros_like(landmarks_gt_positive["labels"])
                
                pred_positive_batch.append(landmarks_pred_positive)
                gt_positive_batch.append(landmarks_gt_positive)

                ious = box_iou(box_convert(landmarks_gt["boxes"], in_fmt="cxcywh", out_fmt="xyxy"),
                               box_convert(landmarks_pred["boxes"], in_fmt="cxcywh", out_fmt="xyxy"))
                
                ious_mask = ious >= 0.3

                labels_correct = T.zeros_like(ious)
                for j, gt_label in enumerate(landmarks_gt["labels"]):
                    labels_correct[j,:] = landmarks_pred["labels"] == gt_label

                both_correct = T.logical_and(ious_mask, labels_correct)
                precision = T.sum(both_correct) / T.sum(ious_mask) if T.sum(ious_mask) != 0 else 0

                only_classification_metric = only_classification_metric*n/(n+1) + precision/(n+1)
                n+=1

            self.metric.update(
                preds=pred_batch, 
                target=gt_batch
            )

            self.metric_only_regression.update(
                preds=pred_positive_batch,
                target=gt_positive_batch
            )

        self.metric.set_dtype(T.float16)
        self.metric.detection_labels = [d.to(T.int16) for d in self.metric.detection_labels]
        self.metric.groundtruth_labels = [d.to(T.int16) for d in self.metric.groundtruth_labels]

        self.metric_only_regression.set_dtype(T.float16)
        self.metric_only_regression.detection_labels = [d.to(T.int16) for d in self.metric_only_regression.detection_labels]
        self.metric_only_regression.groundtruth_labels = [d.to(T.int16) for d in self.metric_only_regression.groundtruth_labels]

        result = {
            self.prefix + k: v
            for k, v in self.metric.compute().items()
        }
        self.metric.reset()

        result_regression = {
            self.prefix + "regression_" + k: v
            for k, v in self.metric_only_regression.compute().items()
        }
        self.metric_only_regression.reset()

        result.update(result_regression)
        result.update({"classification_precision": only_classification_metric})

        return result