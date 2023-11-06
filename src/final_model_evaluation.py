import os
import pickle

import torch as T
import yaml
from tqdm import tqdm
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

    if os.getcwd() == '/Users/marcelloceresini/github/FSOD_CenterNet':
        os.chdir('/Users/marcelloceresini/github/FSOD_CenterNet/src')

    settings = './settings/model_testing_debug_evaluate.yaml'
        
    config = load_settings(settings)

    os.makedirs(config['training']['save_training_info_dir'], exist_ok=True)

    debug_mode = config['debug']['debug_mode_active']
    device = config['device']

    # Dataset generator. Only one of these has to be instantiated. It always returns
    dataset_gen = DatasetsGenerator(config)

    # Instantiate the model (only the base part)
    model = Model(config, n_base_classes=len(dataset_gen.train_base.cats))

    model = model.to(device)
    model.load_state_dict(T.load('../data/weights/from_server/best_model_fix_stoic.pt', map_location="cpu"))

    # Use the dataset generator to generate the base set
    dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
        config['training']['batch_size'], config['training']['num_workers'],
        config['training']['pin_memory'], config['training']['drop_last'], 
        shuffle=True
    )

    print("Before Evaluate")

    metrics_test = Evaluate(
                model, 
                dataset_base_test, 
                prefix="test/",
                device=device,
                config=config
            )(is_novel=False)

    print("After Evaluate")

    with open(os.path.join(config['training']['save_training_info_dir'], 
                            config['training']['base_stats_save_name']), 'wb') as f:
        pickle.dump(metrics_test, f)
