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

    for i, (input_image, labels, n_detections) in tqdm(enumerate(training_loader), total=len(training_loader)):
        loss = 0
        optimizer.zero_grad()

        gt_reg, gt_heat_base, gt_heat_novel = labels        
        pred_reg, pred_heat_base, pred_heat_novel = model(input_image.to(device))

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
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    avg_loss = running_loss / (i + 1)
    return avg_loss