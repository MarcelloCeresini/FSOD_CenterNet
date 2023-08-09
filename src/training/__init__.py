import os
import torch as T

from datetime import datetime

from .train_one_epoch import train_one_epoch_base
from .losses import heatmap_loss_batched, reg_loss_batched

def set_model_to_train_novel(model):
    
    for module in model.named_children():
        if module[0] != "head_novel_heatmap":
            module[1].requires_grad_(False)

    return model


def train_loop_base(model,
                    epochs,
                    training_loader_base,
                    validation_loader_base,
                    optimizer,
                    weights_path,
                    model_name) -> str:

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    
    print("Training base started")

    best_vloss = 1e10
    best_model_path = None

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # train one epoch
        model.train()
        avg_loss = train_one_epoch_base(model,
                                        training_loader_base,
                                        optimizer)

        # validate
        running_vloss = 0.0
        model.eval()

        with T.no_grad():
            for i, (sample, transformed_landmarks, _) in enumerate(validation_loader_base):
                n_detections = [len(l) for l in transformed_landmarks]

                input_image, labels = sample

                gt_reg, gt_heat_base, _ = labels        
                pred_reg, pred_heat_base, _ = model(input_image)

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
            model_path = os.path.join(weights_path, '{}_{}_{}.pt'.format(model_name, timestamp, epoch))
            T.save(model.state_dict(), model_path)
            best_model_path = model_path

    return best_model_path
