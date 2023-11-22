import os
import pickle

import torch as T
import yaml
from tqdm import tqdm
from icecream import ic
import numpy as np

import sys

sys.path.append('/home/volpepe/Desktop/FSOD_CenterNet/src/')
from data_pipeline import DatasetsGenerator
from model import Model
from evaluation import Evaluate


def load_settings(settings_path: str):
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":

    settings = 'settings/base_training.yaml'
        
    config = load_settings(settings)

    device = config['device']

    # Dataset generator. Only one of these has to be instantiated. It always returns
    dataset_gen = DatasetsGenerator(config)

    # Instantiate the model (only the base part)
    model = Model(config, n_base_classes=len(dataset_gen.train_base.cats))

    model.load_state_dict(
        T.load('../data/weights/from_server/best_model_adjusted.pt', 
               map_location="cpu"))
    model = model.to(device)

    # Use the dataset generator to generate the base set
    dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
        config['training']['batch_size'], config['training']['num_workers'],
        config['training']['pin_memory'], config['training']['drop_last'], 
        shuffle=True
    )
    
    # r = np.linspace(0, 0.3, 30)

    # metrics_map     = {c: [] for c in r}
    # metrics_recall  = {c: [] for c in r}
    # mean_n_predictions = {c: [] for c in r}

    # for i in range(20):
    #     for confidence_thresholds in r:
    metrics_test = Evaluate(
        model, 
        dataset_base_test, 
        prefix="test/",
        device=device,
        config=config,
        confidence_threshold=config['eval']['threshold_classification_scores'],
        class_metrics=False
    )(is_novel=False)

    with open('metrics.pickle', 'wb') as f:
        pickle.dump(metrics_test, f)
    #         metrics_map[confidence_thresholds].append(metrics_val['val/map'].item())
    #         metrics_recall[confidence_thresholds].append(metrics_val['val/mar_10'].item())
    #         mean_n_predictions[confidence_thresholds].append(metrics_val['val/mean_n_predictions'])

    # with open('metrics.pickle', 'wb') as f:
    #     pickle.dump({'map': metrics_map, 
    #                  'recall': metrics_recall,
    #                  'mean_n_predictions': mean_n_predictions}, f)
   