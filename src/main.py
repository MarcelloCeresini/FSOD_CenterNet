import argparse
import os
import yaml

import torch as T
from torch.optim import Adam

from data_pipeline import DatasetsGenerator
from evaluation import Evaluate
from model import Model
from training import train_loop_base, set_model_to_train_novel

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
    val_K = conf['data']['K']

    if isinstance(K, list):
        # If K is a list of Ks to try, we should retry the following experiments multiple times (?)
        # Note: in that case we should assume that val_K is a list too.
        raise NotImplementedError

    # base dataset
    dataset_gen = DatasetsGenerator(
        annotations_path = conf['paths']['annotations_path'],
        images_dir = conf['paths']['images_dir'],
        novel_class_ids_path = conf['paths']['novel_classes_ids_path'],
        use_fixed_sets = conf['data']['use_fixed_sets'],
        train_set_path = conf['data']['train_annotations_path'],
        val_set_path = conf['data']['val_annotations_path'],
        K = K, 
        val_K = val_K,
        num_base_classes = conf['data']['base_classes'],
        num_novel_classes = conf['data']['novel_classes'],
        novel_classes_list = conf['data']['novel_classes_list'],
        novel_train_set_path = conf['data']['val_novel_annotations_path'],
        novel_val_set_path = conf['paths']['train_novel_annotations_path'],
    )

    dataset_base_train, dataset_novel_train, \
    dataset_base_val, dataset_novel_val = dataset_gen.generate_datasets()

    model = Model(  encoder_name = conf['model']['encoder_name'], 
                    n_base_classes = conf['data']['base_classes'],
                    n_novel_classes = conf['data']['novel_classes'],
                    head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'],
                    head_novel_heatmap_mode = conf['model']['head_novel_heatmap_mode'])
    model = model.to(device)

    if debug_mode:
        print("Dataset base train length: ", len(dataset_base_train))
        sample, landmarks, original_image_size = dataset_base_train[0]
        # use "show_images.py" functions to show the sample / samples

    optimizer_base = Adam(model.parameters(), lr=conf['training']['base']['lr'])
    
    weights_path = train_loop_base(model,
                                   epochs=conf['training']['base']['epochs'],
                                   training_loader=dataset_base_train,
                                   validation_loader=dataset_base_val,
                                   optimizer=optimizer_base,
                                   name="standard_model_base")

    # copy the weights of the first convolution from the first conv from the base head to the novel head
    with T.no_grad(): 
        model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
        model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

    T.save(model.state_dict(), conf.weights_path) # TODO: decide a path

    # evaluation on base_dataset
    metrics_base = Evaluate(model, dataset_base_test)
    
    # train and eval novel
    metrics_novel_list = []
    for i, (dataset_novel_train, dataset_novel_val, dataset_novel_test) in enumerate(dataset_novel_list):
        print(f"Training on novel dataset n°{i} out of {len(dataset_novel_list)}")
        model.load_state_dict(T.load(weights_path))

        # freeze the weights of everything except the novel head
        model = set_model_to_train_novel(model)



        # evaluation on novel_dataset
        metrics_novel = Evaluate(model, dataset_novel_test)

        # aggregation and print results
        metrics_full = Evaluate(model, dataset_full_test)
    
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