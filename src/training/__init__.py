import torch as T
import numpy as np
from tqdm import tqdm

import datetime

from .train_one_epoch import train_one_epoch
from .losses import heatmap_loss_batched, reg_loss_batched

def set_model_to_train_novel(model, conf):
    '''
    Loads the correct weights and sets the model to train the novel head only
    '''

    model.load_state_dict(T.load(conf['training']['save_base_weights_dir']))

    for module in model.named_children():
        if module[0] != "head_novel_heatmap":
            module[1].requires_grad_(False)

    return model


def train_loop(model,
               epochs,
               training_loader,
               validation_loader,
               optimizer,
               name="standard_model",
               novel_training=False):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    
    if novel_training:
        print("Training novel started")
    else:
        print("Training base started")

    best_vloss = np.inf

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # train one epoch
        # TODO: check if model.train() unfreezes the layers during novel training

        model.train()
        avg_loss = train_one_epoch(model,
                                   training_loader,
                                   optimizer,
                                   novel_training=novel_training)

        # validate
        running_vloss = 0.0
        model.eval()

        with T.no_grad():
            for i, (sample, transformed_landmarks, _) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
                
                n_detections = [len(l) for l in transformed_landmarks]

                input_image, labels = sample

                gt_reg, gt_heat_base, gt_heat_novel = labels        
                pred_reg, pred_heat_base, pred_heat_novel = model(input_image)

                if novel_training:
                    vloss = T.mean(heatmap_loss_batched(pred_heat_novel,
                                                        gt_heat_novel,
                                                        n_detections),
                                   dim=0)
                
                else:
                    vloss = T.mean(heatmap_loss_batched(pred_heat_base,
                                                        gt_heat_base,
                                                        n_detections),
                                dim=0)
            
                    vloss += T.mean(reg_loss_batched(pred_reg,
                                                    gt_reg,
                                                    n_detections),
                                    dim=0)
                
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print('LOSS train {} - valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            model_path = '{}_{}_{}'.format(name, timestamp, epoch)
            
            if not novel_training:
                T.save(model.state_dict(), model_path)

    return model
