import os
from typing import Dict
import torch as T
from tqdm import tqdm
import wandb

from model import Model
from .train_one_epoch import train_one_epoch
from .losses import heatmap_loss_batched, reg_loss_batched

# CALLBACKS: 
# - reduce learning rate on plateau
# - early stopping
# - save best model
# - log to w&b

def set_model_to_train_novel(model: Model, conf: Dict):
    '''
    Loads the correct weights and sets the model to train the novel head only
    '''
    # Load the weights from the base training
    model.load_state_dict(
        T.load(conf['training']['save_base_weights_dir'] + 'best_base.pt', map_location='cpu'),
        strict=False
    )

    # Add the base head weights to the novel head
    with T.no_grad(): 
        model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
        model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

    # Finally, stop training for base model (only novel model shall be learned)
    for module in model.named_children():
        if module[0] != "head_novel_heatmap":
            module[1].requires_grad_(False)

    return model


def train_loop(model,
               epochs,
               training_loader,
               validation_loader,
               optimizer,
               weights_path=None,
               name="standard_model",
               novel_training=False):
    
    print("####################")
    print(f"Started {'base' if not novel_training else 'novel'} training")
    print("####################")

    best_vloss = 1e10

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        # Train for one epoch
        # TODO: does model.train() unfreeze the base weights while novel training?

        model.train()
        avg_loss = train_one_epoch(model,
                                   training_loader,
                                   optimizer,
                                   novel_training=novel_training)
        
        

        # Validate
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
            
                    # TODO: shouldnt the reg loss be computed on the novel head as well?
                    vloss += T.mean(reg_loss_batched(pred_reg,
                                                    gt_reg,
                                                    n_detections),
                                    dim=0)
                
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        # TODO: add some metric?

        print('LOSS train {} - valid {}'.format(avg_loss, avg_vloss))
        # TODO: Log the running loss averaged per batch for both training and validation
        wandb.log({"epoch": epoch, 
                   "loss": avg_loss,
                   "val_loss": avg_vloss
        })

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            if not novel_training:
                model_path = os.path.join(weights_path, 'best_base.pt')
                T.save(model.state_dict(), model_path)

    return model
