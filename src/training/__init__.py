import os
from typing import Dict
import torch as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import Model
from training.callbacks import EarlyStopper
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
        module[1].requires_grad_(module[0] == "head_novel_heatmap")

    return model


def setup_warm_start(model: Model, optimizer, freeze=True):
    # Stop training for encoder
    for module in model.named_children():
        if 'encoder' in module[0]:
            module[1].requires_grad_(not freeze)
        else:
            module[1].requires_grad_(True)

    # Reduce learning rate
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] * (0.01 if freeze else 100) 


def train_loop(model,
               config,
               training_loader:DataLoader,
               validation_loader:DataLoader,
               optimizer,
               scheduler,
               device,
               novel_training=False,
               novel_k=None):
    
    train_group = 'base' if not novel_training else 'novel'
    epochs = config['training'][train_group]['epochs']
    weights_path = config['training']['save_base_weights_dir']
    epoch_metric_log_interval = config['training'][train_group]['epoch_metric_log_interval'] / \
        (novel_k if novel_k is not None else 1)
    
    # print("####################")
    # print(f"Started {'base' if not novel_training else 'novel'} training")
    # print("####################")

    best_vloss = 1e10
    batch_count = 0

    early_stopper = EarlyStopper(
        patience=config['training']['base' if not novel_training else 'novel']['early_stopping_patience'] / \
            (novel_k if novel_k is not None else 1),
        min_delta=config['training']['base' if not novel_training else 'novel']['early_stopping_min_delta']
    )

    if config['training']['use_class_weights']:
        train_base_coco_dset = training_loader.dataset.coco
        cats_list = list(train_base_coco_dset.cats)
        ann_cat_counts = T.zeros(len(cats_list))
        # Compute class counts for annotations
        for class_id in train_base_coco_dset.cats:
            ann_cat_counts[cats_list.index(class_id)] = len(train_base_coco_dset.getAnnIds(catIds=[class_id]))
        # Weights are inverse frequencies
        class_weights = T.stack(
            [T.full(
                (int(config["data"]["input_to_model_resolution"][0]/config["data"]["output_stride"][0]),
                int(config["data"]["input_to_model_resolution"][1]/config["data"]["output_stride"][1])),
             T.sum(ann_cat_counts) / cat_count)
            for cat_count in ann_cat_counts]).to(device)
    else:
        class_weights = None

    for epoch in tqdm(range(epochs), 
                      desc=f"{'Base' if not novel_training else 'Novel'} Training Epochs: ",
                      position=0 + int(novel_training),
                      leave=not novel_training):

        if not novel_training and config['training']['base']['warm_start'] and epoch == 0:
            setup_warm_start(model, optimizer)
        elif not novel_training and config['training']['base']['warm_start'] and epoch == 1:
            setup_warm_start(model, optimizer, freeze=False)
        elif novel_training and config['training']['novel']['warm_start'] and epoch == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.001
        elif novel_training and config['training']['novel']['warm_start'] and epoch == \
            int(40/(novel_k if novel_k is not None else 1)):
            for g in optimizer.param_groups:
                g['lr'] *= 1000

        # Train for one epoch
        model.train()
        avg_loss, batch_count = train_one_epoch(model,
                                                config,
                                                training_loader,
                                                optimizer,
                                                device,
                                                novel_training=novel_training,
                                                batch_count=batch_count,
                                                class_weights=class_weights)

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
                                                config, class_weights),
                                   dim=0)
                else:
                    vloss_heat = T.mean(heatmap_loss(pred_heat_base,
                                                gt_heat_base,
                                                n_detections,
                                                config, class_weights),
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
            "lr": optimizer.param_groups[0]['lr'],
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
                config=config,
                confidence_threshold=config['eval']['threshold_classification_scores']
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

        # Early stopping
        if early_stopper.early_stop(avg_vloss):
            break

    return model
