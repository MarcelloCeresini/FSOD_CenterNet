import torch as T

def heatmap_loss(pred_heatmap, gt_heatmap, num_keypoints, config, weights = None):
    # TODO: num_keypoints is not coherent with gt_heatmap == 1
    gt_heatmap = gt_heatmap.to(pred_heatmap.device)
    num_keypoints = num_keypoints.to(pred_heatmap.device)

    if weights is not None:
        loss = T.where(
            gt_heatmap == 1,
            (1 - pred_heatmap) ** config['model']['alpha_loss'] * T.log(pred_heatmap) * weights,
            (1 - gt_heatmap) ** config['model']['beta_loss'] * \
                (pred_heatmap) ** config['model']['alpha_loss'] * T.log(1 - pred_heatmap) * weights,
        ).reshape(pred_heatmap.shape[0], -1).sum(dim=-1)

        loss /= T.sum(weights[:,0,0]) * weights.shape[0]

    else:
        loss = T.where(
            gt_heatmap == 1,
            (1 - pred_heatmap) ** config['model']['alpha_loss'] * T.log(pred_heatmap),
            (1 - gt_heatmap) ** config['model']['beta_loss'] * \
                (pred_heatmap) ** config['model']['alpha_loss'] * T.log(1 - pred_heatmap),
        ).reshape(pred_heatmap.shape[0], -1).sum(dim=-1)

    result = T.where(
        num_keypoints != 0,
        input = loss / num_keypoints,
        other = 0
    )

    return -result


def reg_loss(pred_reg, gt_reg, num_keypoints, config):
    '''
    ```
    regressor_label[0, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][0]
    regressor_label[1, lr_cp_idx[0], lr_cp_idx[1]] = l["size"][1]
    regressor_label[2, lr_cp_idx[0], lr_cp_idx[1]] = offset[0]
    regressor_label[3, lr_cp_idx[0], lr_cp_idx[1]] = offset[1]
    ```
    '''
    gt_reg = gt_reg.to(pred_reg.device)
    num_keypoints = num_keypoints.to(pred_reg.device)

    pred_size, gt_size = pred_reg[:, 0:2], gt_reg[:, 0:2]
    pred_offset, gt_offset = pred_reg[:, 2:4], gt_reg[:, 2:4]

    loss_size = T.where(gt_size != 0., T.abs(pred_size - gt_size), 0.)\
        .reshape(pred_size.shape[0], -1).sum(dim=-1)
    
    loss_offset = T.where(gt_offset != 0., T.abs(pred_offset - gt_offset), 0.)\
        .reshape(pred_offset.shape[0], -1).sum(dim=-1)
    
    loss = (loss_size*config['model']['lambda_size_loss'] + \
            loss_offset*config['model']['lambda_offset_loss'])
    
    result = T.where(
        num_keypoints != 0.,
        input = loss / num_keypoints,
        other = 0.
    )

    return result
