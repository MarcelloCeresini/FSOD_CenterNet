import torch as T
from tqdm import tqdm

from .losses import heatmap_loss, reg_loss

def train_one_epoch(model,
                    training_loader,
                    optimizer,
                    device,
                    novel_training=False,
                    # epoch_index, 
                    # tb_writer,
                    ):
    
    running_loss = 0.
    steps = 0

    for i, (input_image, labels, n_detections, _) in tqdm(enumerate(training_loader), total=len(training_loader)):
        loss = 0
        optimizer.zero_grad()

        gt_reg, gt_heat_base, gt_heat_novel = labels
        pred_reg, pred_heat_base, pred_heat_novel = model(input_image.to(device))

        # TODO: example: gt_head_base is a [batch_size, n_base_classes, H, W] tensor
        # how does the loss work with a batch?
        # num keypoints is a list of the number of keypoints for each image in the batch
        # maybe we should "map" each image to its own loss and then do the mean

        if novel_training:
            loss += T.mean(heatmap_loss(
                        pred_heat_novel, gt_heat_novel, n_detections
                    ), dim=0)
            
        else:
            loss += T.mean(heatmap_loss(
                        pred_heat_base, gt_heat_base, n_detections
                    ), dim=0)
            
        loss += T.mean(reg_loss(
                    pred_reg, gt_reg, n_detections
                ), dim=0)
        
        if loss > 0.:
            steps += 1
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # TODO: THIS IS ONLY FOR TESTING
        if steps > 2:
            break


    avg_loss = running_loss / steps
    return avg_loss