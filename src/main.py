import torch as T
import yaml
from torch.optim import Adam

import argparse
import os
import pickle

from data_pipeline import DatasetsGenerator
from evaluation import Evaluate
from model import Model
from training import set_model_to_train_novel, train_loop


def parse_args():
    arps = argparse.ArgumentParser()
    arps.add_argument('-sett', '--settings', type=str, help='Settings YAML file')
    return arps.parse_args()


def load_settings(settings_path: str):
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):

    conf = load_settings(args.settings)
    debug_mode = conf['debug']['debug_mode_active']
    device = conf['device']

    K = conf['data']['K']
    val_K = conf['data']['val_K']
    test_K = conf['data']['test_K']
    n_repeats_novel_train = conf['training']['repeat_novel_training']

    if isinstance(K, int):
        K = [K]
    if isinstance(val_K, int):
        val_K = [val_K]
    if isinstance(test_K, int):
        test_K = [test_K]

    # Dataset generator. Only one of these has to be instantiated. It always returns
    dataset_gen = DatasetsGenerator(
        annotations_path = conf['paths']['annotations_path'],
        images_dir = conf['paths']['images_dir'],
        novel_class_ids_path = conf['paths']['novel_classes_ids_path'],
        train_set_path = conf['paths']['train_base_annotations_path'],
        val_set_path = conf['paths']['val_base_annotations_path'],
        test_set_path = conf['paths']['test_base_annotations_path'],
        use_fixed_novel_sets = conf['data']['use_fixed_sets'],
        novel_train_set_path = conf['paths']['train_novel_annotations_path'],
        novel_val_set_path = conf['paths']['val_novel_annotations_path'],
        novel_test_set_path = conf['paths']['test_novel_annotations_path']
    )

    # Use the dataset generator to generate the base set
    dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
        conf['training']['batch_size'], conf['training']['num_workers'],
        conf['training']['pin_memory'], conf['training']['drop_last'], shuffle=True
    )

    # Instantiate the model
    model = Model(  encoder_name = conf['model']['encoder_name'], 
                    n_base_classes = len(dataset_gen.train_base.cats),
                    n_novel_classes = len(dataset_gen.novel_classes),
                    head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'],
                    head_novel_heatmap_mode = conf['model']['head_novel_heatmap_mode'])
    model = model.to(device)

    if debug_mode:
        print("Dataset base train length: ", len(dataset_base_train))
        # sample, landmarks, original_image_size = dataset_base_train[0]
        # use "show_images.py" functions to show the sample / samples

    optimizer_base = Adam(model.parameters(), 
                          lr=conf['training']['base']['lr'])
    
    # Train the base model
    # TODO: IMAGES ARE MISSING!
    model = train_loop(model,
                       epochs=conf['training']['base']['epochs'],
                       training_loader_base=dataset_base_train,
                       validation_loader_base=dataset_base_val,
                       optimizer=optimizer_base,
                       model_name="standard_model_base")

    with T.no_grad(): 
        model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
        model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

    T.save(model.state_dict(), 
           conf['training']['save_base_weights_dir'])
    
    # Evaluation on base test dataset
    metrics_base = Evaluate(model, 
                            dataset_base_test)
    
    with open(os.path.join(conf['training']['save_training_info_dir'], 'base_training_info.pkl'), 'wb') as f:
        pickle.dump(metrics_base, f)

    ## NOVEL TRAININGS ##

    # Train and eval novel
    metrics_novel_list = {k: [] for k in conf['data']['K']}
    metrics_full_list = {k: [] for k in conf['data']['K']}

    total_trainings = len(K) * n_repeats_novel_train
    for i in range(total_trainings):
        print(f"\nTraining on novel dataset n°{i} out of {total_trainings}")

        current_train_K = K[i % n_repeats_novel_train]
        current_val_K   = val_K[i % n_repeats_novel_train]
        current_test_K  = test_K[i % n_repeats_novel_train]

        model = set_model_to_train_novel(model, conf)

        # Optimizer
        optimizer_novel = Adam(model.parameters(), 
                               lr=conf['training']['novel']['lr'])

        # Obtain dataset from generator
        # Use the dataset generator to generate the base set
        _, (dataset_novel_train, dataset_novel_val, dataset_novel_test) = \
            dataset_gen.generate_dataloaders(
                K=current_train_K, val_K=current_val_K, test_K=current_test_K,
                num_novel_classes_to_sample=conf['data']['novel_classes_to_sample'],
                novel_classes_to_sample_list=conf['data']['novel_classes_list'],
                gen_random_seed=conf['data']['gen_random_seed'],
                batch_size=conf['training']['batch_size'],
                num_workers=conf['training']['num_workers'],
                pin_memory=conf['training']['pin_memory'],
                drop_last=conf['training']['drop_last'],
                shuffle=True
            )

        model = train_loop(model,
                           epochs=conf.epochs_novel,
                           training_loader=dataset_novel_train,
                           validation_loader=dataset_novel_val,
                           optimizer=optimizer_novel,
                           novel_training=True)
        
        # Evaluation on novel_dataset
        metrics_novel = Evaluate(model, dataset_novel_test)
        metrics_novel_list[current_train_K].append(metrics_novel)

        # TODO: merge the two datasets
        # Evaluation on full dataset
        metrics_full = Evaluate(model, dataset_full_test)
        metrics_full_list[current_train_K].append(metrics_full)
    
    with open(os.path.join(conf['training']['save_training_info_dir'], 'novel_training_info.pkl'), 'wb') as f:
        pickle.dump(metrics_novel_list, f)
    
    with open(os.path.join(conf['training']['save_training_info_dir'], 'full_training_info.pkl'), 'wb') as f:
        pickle.dump(metrics_full_list, f)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)