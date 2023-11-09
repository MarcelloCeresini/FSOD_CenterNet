import argparse
import os
import pickle

import torch as T
import yaml
from torch.optim import Adam
from tqdm import tqdm
from time import time
from icecream import ic

import sys

sys.path.append('/Users/marcelloceresini/github/FSOD_CenterNet/src')
from data_pipeline import DatasetsGenerator
from model import Model
from evaluation import Evaluate


def load_settings(settings_path: str):
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    if os.getcwd() == "/Users/marcelloceresini/github/FSOD_CenterNet":
        os.chdir("/Users/marcelloceresini/github/FSOD_CenterNet/src")

    settings = './settings/base_training.yaml'
        
    config = load_settings(settings)

    device = config['device']

    dataset_gen = DatasetsGenerator(config)

    if config['training']['train_base']:

        # Instantiate the model (only the base part)
        model = Model(config, n_base_classes=len(dataset_gen.train_base.cats))
        model = model.to(device)

        # Use the dataset generator to generate the base set
        dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
            config['training']['batch_size'], config['training']['num_workers'],
            config['training']['pin_memory'], config['training']['drop_last'], 
            shuffle=True
        )

        ann_cat_counts = T.zeros(len(dataset_gen.train_base.cats))
        for class_id in dataset_gen.train_base.cats:
            ann_cat_counts[list(dataset_gen.train_base.cats).index(class_id)] = len(dataset_gen.train_base.getAnnIds(catIds=[class_id]))

        weights = T.stack(
            [T.full(
                (int(config["data"]["input_to_model_resolution"][0]/config["data"]["output_stride"][0]),
                int(config["data"]["input_to_model_resolution"][1]/config["data"]["output_stride"][1])),
             T.sum(ann_cat_counts) / cat_count)
            for cat_count in ann_cat_counts])

    tic = time()
    metrics_test = Evaluate(
            model, 
            dataset_base_test, 
            prefix="test/",
            device=device,
            config=config,
        )(is_novel=False)

    ic("Standard", time()-tic)

    tic = time()
    metrics_test = Evaluate(
            model, 
            dataset_base_test, 
            prefix="test/",
            device=device,
            config=config,
            more_metrics=True
        )(is_novel=False)

    ic("More metrics", time()-tic)

    tic = time()
    metrics_test = Evaluate(
            model, 
            dataset_base_test, 
            prefix="test/",
            device=device,
            config=config,
            half_precision=True
        )(is_novel=False)

    ic("Half precision", time()-tic)

    tic = time()
    metrics_test = Evaluate(
            model, 
            dataset_base_test, 
            prefix="test/",
            device=device,
            config=config,
            more_metrics=True,
            half_precision=True
        )(is_novel=False)

    ic("More metrics + half precision", time()-tic)