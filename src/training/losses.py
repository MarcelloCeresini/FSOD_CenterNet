import torch as T
from torch.func import vmap

from model.config import Config

conf = Config()

def heatmap_loss(pred_heatmap, gt_heatmap, num_keypoints):
    # TODO: num_keypoints is not coherent with gt_heatmap == 1
    gt_heatmap = gt_heatmap.to(pred_heatmap.device)
    num_keypoints = num_keypoints.to(pred_heatmap.device)

    loss = T.where(
        gt_heatmap == 1,
        (1 - pred_heatmap) ** conf.alpha_loss * T.log(pred_heatmap),
        (1 - gt_heatmap) ** conf.beta_loss * (pred_heatmap) ** conf.alpha_loss * T.log(1 - pred_heatmap),
    ).reshape(pred_heatmap.shape[0], -1).sum(dim=-1)

    return -loss / num_keypoints


def reg_loss(pred_reg, gt_reg, num_keypoints):
    '''
    regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][0]
    regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][1]
    regressor_label[2, lr_cp_idx[0], lr_cp_idx[1]] = offset[0]
    regressor_label[3, lr_cp_idx[0], lr_cp_idx[1]] = offset[1]
    '''
    gt_reg = gt_reg.to(pred_reg.device)
    num_keypoints = num_keypoints.to(pred_reg.device)

    pred_size, gt_size = pred_reg[:, 0:2], gt_reg[:, 0:2]
    pred_offset, gt_offset = pred_reg[:, 2:4], gt_reg[:, 2:4]

    loss_size = T.where(gt_size != 0., T.abs(pred_size - gt_size), 0.)\
        .reshape(pred_size.shape[0], -1).sum(dim=-1)
    
    loss_offset = T.where(gt_offset != 0., T.abs(pred_offset - gt_offset), 0.)\
        .reshape(pred_offset.shape[0], -1).sum(dim=-1)
    
    return (loss_size*conf.lambda_size_loss + \
            loss_offset*conf.lambda_offset_loss) / num_keypoints
