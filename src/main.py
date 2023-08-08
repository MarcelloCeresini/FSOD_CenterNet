import argparse
import os
import pickle

import torch as T
import yaml
from torch.optim import Adam

from data_pipeline import DatasetsGenerator
from evaluation import Evaluate
from model import Model
from training import set_model_to_train_novel, train_loop_base


def parse_args():
    arps = argparse.ArgumentParser()
    arps.add_argument('-sett', '--settings', type='str', help='Settings YAML file')
    return arps.parse_args()


def load_settings(settings_path: str):
    with open(settings_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):

    conf = load_settings(args.settings)
    debug_mode = conf['debug']['debug_mode_active']
    device = conf['device']

    assert (not os.path.exists(conf['training']['save_weights_path'])), \
        f"Weights path {conf['training']['save_weights_path']} already exists, will not overwrite, so delte it first or change the path."

    K = conf['data']['K']
    val_K = conf['data']['val_K']
    test_K = conf['data']['test_K']
    n_repeats_novel_train = conf['data']['repeat_novel_training']

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
        train_set_path = conf['data']['train_annotations_path'],
        val_set_path = conf['data']['val_annotations_path'],
        test_set_path = conf['data']['test_annotations_path'],
        use_fixed_sets = conf['data']['use_fixed_sets'],
        novel_classes_list = conf['data']['novel_classes_list'],
        novel_train_set_path = conf['data']['val_novel_annotations_path'],
        novel_val_set_path = conf['paths']['train_novel_annotations_path'],
    )

    # Use the dataset generator to generate the base set
    dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
        conf['training']['batch_size'], conf['training']['num_workers'],
        conf['training']['pin_memory'], conf['training']['drop_last'], shuffle=True
    )

    # Instantiate the model
    model = Model(  encoder_name = conf['model']['encoder_name'], 
                    n_base_classes = len(dataset_gen.train_base.cats),
                    n_novel_classes = len(dataset_gen.train_novel.cats),
                    head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'],
                    head_novel_heatmap_mode = conf['model']['head_novel_heatmap_mode'])
    model = model.to(device)

    if debug_mode:
        print("Dataset base train length: ", len(dataset_base_train))
        sample, landmarks, original_image_size = dataset_base_train[0]
        # use "show_images.py" functions to show the sample / samples

    optimizer_base = Adam(model.parameters(), lr=conf['training']['base']['lr'])
    
    # Train the base model
    best_base_weights = train_loop_base(model,
        epochs=conf['training']['base']['epochs'],
        training_loader=dataset_base_train,
        validation_loader=dataset_base_val,
        optimizer=optimizer_base,
        weights_path=conf['training']['save_base_weights_path'],
        name="standard_model_base")

    # Evaluation on base test dataset
    metrics_base = Evaluate(model, dataset_base_test)
    
    with open(os.path.join(conf['training']['save_training_info_dir'], 'base_training_info.pkl'), 'wb') as f:
        pickle.dump(metrics_base, f)

    ## NOVEL TRAININGS ##

    # Train and eval novel
    metrics_novel_list = []
    total_trainings = len(K) * n_repeats_novel_train
    for i in range(total_trainings):
        print(f"\nTraining on novel dataset n°{i} out of {total_trainings}")

        current_train_K = K[i % n_repeats_novel_train]
        current_val_K   = val_K[i % n_repeats_novel_train]
        current_test_K  = test_K[i % n_repeats_novel_train]

        # Load weights from base model
        model.load_state_dict(T.load(best_base_weights))
        # Copy the weights of the first convolution from the first conv from the base head to the novel head
        with T.no_grad(): 
            model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
            model.head_novel_heatmap.conv1.bias.data   = model.head_base_heatmap.conv1.bias.data
        # Freeze the weights of everything except the novel head
        model = set_model_to_train_novel(model)

        # Optimizer
        optimizer_novel = Adam(model.parameters(), lr=conf['training']['novel']['lr'])

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

        best_novel_weights_for_params = train_loop_base(model,
            epochs=conf['training']['novel']['epochs'],
            training_loader=dataset_novel_train,
            validation_loader=dataset_novel_val,
            optimizer=optimizer_novel,
            weights_path=conf['training']['save_novel_weights_dir']
            name=f"novel_model_K_{current_train_K}_{i // n_repeats_novel_train}")
        
        # Evaluation on novel_dataset
        metrics_novel = Evaluate(model, dataset_novel_test)
        metrics_novel_list.append(metrics_novel)
    
    with open(os.path.join(conf['training']['save_training_info_dir'], 'novel_training_info.pkl'), 'wb') as f:
        pickle.dump(metrics_novel_list, f)
    
    '''
    - ``map_dict``: A dictionary containing the following key-values:

        - map: (:class:`~torch.Tensor`), global mean average precision
        - map_small: (:class:`~torch.Tensor`), mean average precision for small objects
        - map_medium:(:class:`~torch.Tensor`), mean average precision for medium objects
        - map_large: (:class:`~torch.Tensor`), mean average precision for large objects
        - mar_1: (:class:`~torch.Tensor`), mean average recall for 1 detection per image
        - mar_10: (:class:`~torch.Tensor`), mean average recall for 10 detections per image
        - mar_100: (:class:`~torch.Tensor`), mean average recall for 100 detections per image
        - mar_small: (:class:`~torch.Tensor`), mean average recall for small objects
        - mar_medium: (:class:`~torch.Tensor`), mean average recall for medium objects
        - mar_large: (:class:`~torch.Tensor`), mean average recall for large objects
        - map_50: (:class:`~torch.Tensor`) (-1 if 0.5 not in the list of iou thresholds), mean average precision at
          IoU=0.50
        - map_75: (:class:`~torch.Tensor`) (-1 if 0.75 not in the list of iou thresholds), mean average precision at
          IoU=0.75
        - map_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average precision per
          observed class
        - mar_100_per_class: (:class:`~torch.Tensor`) (-1 if class metrics are disabled), mean average recall for 100
          detections per image per observed class
        - classes (:class:`~torch.Tensor`), list of all observed classes
    '''

if __name__ == "__main__":
    args = parse_args()
    main(args)