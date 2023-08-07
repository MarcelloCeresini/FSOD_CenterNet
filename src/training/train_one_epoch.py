import torch as T
from tqdm import tqdm

from losses import heatmap_loss_batched, reg_loss_batched

def train_one_epoch_base(model,
                         training_loader_base,
                         optimizer,
                         # epoch_index, 
                         # tb_writer,
                         ):
    
    running_loss = 0.

    for i, (sample, transformed_landmarks, _) in tqdm(enumerate(training_loader_base), total=len(training_loader_base)):
        loss = 0
        optimizer.zero_grad()

        n_detections = [len(l) for l in transformed_landmarks]

        input_image, labels = sample

        gt_reg, gt_heat_base, _ = labels        
        pred_reg, pred_heat_base, _ = model(input_image)

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


def train_one_epoch_novel(model,
                          training_loader_novel,
                          optimizer,
                          # epoch_index, 
                          # tb_writer,
                          ):
    
    running_loss = 0.

    for i, (sample, transformed_landmarks, _) in tqdm(enumerate(training_loader_novel)):
        loss = 0
        optimizer.zero_grad()

        n_detections = [len(l) for l in transformed_landmarks]

        input_image, labels = sample

        gt_reg, _, gt_heat_novel = labels        
        pred_reg, _, pred_heat_novel = model(input_image)

        loss += T.mean(heatmap_loss_batched(pred_heat_novel,
                                            gt_heat_novel,
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