import torch as T
from torch.func import vmap

from model.config import Config

conf = Config()

def heatmap_loss(pred_heatmap, gt_heatmap, num_keypoints):
    
    loss = T.sum(T.where(gt_heatmap==1,
                         (1-pred_heatmap)**conf.alpha_loss * T.log(pred_heatmap), 
                         (1-gt_heatmap)**conf.beta_loss * (pred_heatmap)**conf.alpha_loss * T.log(1-pred_heatmap)))
    
    return -loss / num_keypoints
    
def heatmap_loss_batched():
    return vmap(heatmap_loss)


def reg_loss(pred_reg, gt_reg, num_keypoints):
    '''
    regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][0]
    regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][1]
    regressor_label[2, lr_cp_idx[0], lr_cp_idx[1]] = offset[0]
    regressor_label[3, lr_cp_idx[0], lr_cp_idx[1]] = offset[1]
    '''
    pred_size, gt_size = pred_reg[0:2], gt_reg[0:2]
    pred_offset, gt_offset = pred_reg[2:4], gt_reg[2:4]

    loss_size = T.sum(T.where(gt_size,
                              T.abs(pred_size - gt_size),
                              0))
    
    loss_offset = T.sum(T.where(gt_offset,
                                T.abs(pred_offset - gt_offset),
                                0))
    
    return (loss_size*conf.lambda_size_loss + loss_offset*conf.lambda_offset_loss) / num_keypoints

def reg_loss_batched():
    return vmap(reg_loss)