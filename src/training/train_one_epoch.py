import torch as T
from tqdm import tqdm

from .losses import heatmap_loss, reg_loss

def train_one_epoch(model,
                    config,
                    training_loader,
                    optimizer,
                    device,
                    novel_training=False):
    
    running_loss = 0.

    for i, (input_image, labels, n_detections, _) in tqdm(enumerate(training_loader), 
                                                          total=len(training_loader),
                                                          position=1 + int(novel_training),
                                                          leave=False,
                                                          desc="Training: "):
        loss = 0
        optimizer.zero_grad()

        gt_reg, gt_heat_base, gt_heat_novel = labels
        pred_reg, pred_heat_base, pred_heat_novel = model(input_image.to(device))

        if novel_training:
            loss_1 = T.mean(heatmap_loss(
                        pred_heat_novel, gt_heat_novel, n_detections, config
                    ), dim=0)
            
        else:
            loss_1 = T.mean(heatmap_loss(
                        pred_heat_base, gt_heat_base, n_detections, config
                    ), dim=0)
            
        loss_2 = T.mean(reg_loss(
                    pred_reg, gt_reg, n_detections, config
                ), dim=0)
        
        loss = loss_1 + loss_2
        
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['loss_clip_norm'])
        optimizer.step()
        running_loss += loss.item()

        # # TODO: THIS IS ONLY FOR TESTING
        # if steps > 2:
        #     break

    avg_loss = running_loss / (i + 1)
    return avg_loss