import os
from typing import Dict
import torch as T
from tqdm import tqdm
import wandb

from model import Model
from .train_one_epoch import train_one_epoch
from .losses import heatmap_loss, reg_loss
from evaluation import Evaluate

# CALLBACKS: 
# - reduce learning rate on plateau
# - early stopping
# - save best model
# - log to w&b

def set_model_to_train_novel(model: Model, config: Dict):
    '''
    Loads the correct weights and sets the model to train the novel head only
    '''
    # Load the weights from the base training
    model.load_state_dict(
        T.load(config['training']['save_base_weights_dir'] + config['training']['base_weights_load_name'], 
               map_location='cpu'),
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
               config,
               training_loader,
               validation_loader,
               optimizer,
               scheduler,
               device,
               novel_training=False):
    
    train_group = 'base' if not novel_training else 'novel'
    epochs = config['training'][train_group]['epochs']
    weights_path = config['training']['save_base_weights_dir']
    epoch_metric_log_interval = config['training'][train_group]['epoch_metric_log_interval']
    
    # print("####################")
    # print(f"Started {'base' if not novel_training else 'novel'} training")
    # print("####################")

    best_vloss = 1e10
    batch_count = 0

    for epoch in tqdm(range(epochs), 
                      desc=f"{'Base' if not novel_training else 'Novel'} Training Epochs: ",
                      position=0 + int(novel_training),
                      leave=not novel_training):

        # Train for one epoch
        model.train()
        avg_loss, batch_count = train_one_epoch(model,
                                                config,
                                                training_loader,
                                                optimizer,
                                                device,
                                                novel_training=novel_training,
                                                batch_count=batch_count)

        # Validate
        running_vloss = 0.0

        model.eval()
        with T.no_grad():
            for i, (input_image, labels, n_detections, _) in tqdm(enumerate(validation_loader), 
                    total=len(validation_loader), position=1 + int(novel_training), leave=False,
                    desc="Validation: "):
                
                gt_reg, gt_heat_base, gt_heat_novel = labels
                pred_reg, pred_heat_base, pred_heat_novel = model(input_image.to(device))

                if novel_training:
                    vloss_heat = T.mean(heatmap_loss(pred_heat_novel,
                                                gt_heat_novel,
                                                n_detections,
                                                config),
                                   dim=0)
                else:
                    vloss_heat = T.mean(heatmap_loss(pred_heat_base,
                                                gt_heat_base,
                                                n_detections,
                                                config),
                                dim=0)

                vloss_reg = T.mean(reg_loss(pred_reg,
                                         gt_reg,
                                         n_detections,
                                         config),
                                dim=0)

                running_vloss += (vloss_heat + vloss_reg)

        avg_vloss = running_vloss.item() / (i + 1)

        log_dict = {
            "epoch": epoch, 
            "train/avg_loss": avg_loss,
            "val/avg_loss": avg_vloss,
            "val/heatmap_loss": vloss_heat,
            "val/reg_loss": vloss_reg
        }

        if epoch % epoch_metric_log_interval == 0:
            metrics_validation = Evaluate(
                model, 
                validation_loader, 
                prefix="val/",
                device=device,
                config=config
            )(is_novel=novel_training)
            log_dict.update(metrics_validation)

        wandb.log(log_dict)

        # LR Scheduling
        scheduler.step(avg_vloss)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            if not novel_training:
                os.makedirs(weights_path, exist_ok=True)
                model_path = os.path.join(weights_path, 
                                          config['training']['base_weights_save_name'])
                T.save(model.state_dict(), model_path)

    return model
