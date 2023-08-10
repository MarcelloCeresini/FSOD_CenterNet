import torch as T
from tqdm import tqdm

from losses import heatmap_loss_batched, reg_loss_batched

def train_one_epoch(model,
                    training_loader,
                    optimizer,
                    novel_training=False,
                    # epoch_index, 
                    # tb_writer,
                    ):
    
    running_loss = 0.

    for i, (sample, transformed_landmarks, _) in tqdm(enumerate(training_loader), total=len(training_loader)):
        loss = 0
        optimizer.zero_grad()

        n_detections = [len(l) for l in transformed_landmarks]

        input_image, labels = sample

        gt_reg, gt_heat_base, gt_heat_novel = labels        
        pred_reg, pred_heat_base, pred_heat_novel = model(input_image)

        if novel_training:
            loss += T.mean(heatmap_loss_batched(pred_heat_novel,
                                                gt_heat_novel,
                                                n_detections),
                           dim=0)
            
        else:
            loss += T.mean(heatmap_loss_batched(pred_heat_base,
                                                gt_heat_base,
                                                n_detections),
                        dim=0)
            
            loss += T.mean(reg_loss_batched(pred_reg,
                                            gt_reg,
                                            n_detections),
                        dim=0)
        
        loss.backward()
        optimizer.step()

        # TODO: which one?
        raise NotImplementedError("still havent decided which one to use")
        running_loss += loss.item()
        running_loss += loss

    return running_loss