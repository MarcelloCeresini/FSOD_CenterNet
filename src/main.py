import argparse
import yaml

import torch as T

from data_pipeline import DatasetsFromCocoAnnotations
from evaluation import Evaluate
from model import Model


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

    K = conf['data']['K']
    val_K = conf['data']['K']

    if isinstance(K, list):
        # If K is a list of Ks to try, we should retry the following experiments multiple times (?)
        # Note: in that case we should assume that val_K is a list too.
        raise NotImplementedError

    # base dataset
    dataset_gen = DatasetsFromCocoAnnotations(
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
    
    # TODO: we need a dataloader for automatic batches/gpu loading

    model = Model(  encoder_name = conf['model']['encoder_name'], 
                    n_base_classes = conf['data']['base_classes'],
                    n_novel_classes = conf['data']['novel_classes'],
                    head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'],
                    head_novel_heatmap_mode = conf['model']['head_novel_heatmap_mode'])
    model = model.to(device)

    # dataset_full_test = DatasetFromCocoAnnotations( annotations_path="full_2017_bboxes.json",
    #                                                 annotations_dir="data",
    #                                                 images_dir="always_the_same_dir",
    #                                                 randomly_selected_novel = False,
    #                                                 transform=TransformTesting())

    if debug_mode:
        print("Dataset base train length: ", len(dataset_base_train))
        sample, landmarks, original_image_size = dataset_base_train[0]
        # use "show_images.py" functions to show the sample / samples


    # first training on base_dataset: loss is ZERO on novel head
    for sample, _, _ in dataset_base_train:
        input_image, labels = sample
        # forward pass
        pass

    # copy the weights of the first convolution from the first conv from the base head to the novel head
    with T.no_grad(): 
        model.head_novel_heatmap.conv1.weight.data = model.head_base_heatmap.conv1.weight.data
        model.head_novel_heatmap.conv1.bias.data = model.head_base_heatmap.conv1.bias.data

    # freeze the weights of everything except the novel head
    for module in model.named_children():
        if module[0] != "head_novel_heatmap":
            module[1].requires_grad_(False)
            module[1].detach()

    # training on novel_dataset: loss on novel head is the one you used for base
    for sample, _, _ in dataset_novel_train:
        # forward pass
        input_image, labels = sample
        pass


    # evaluation on base_dataset
    metrics_base = Evaluate(model,
                            dataset_base_val)

    # evaluation on novel_dataset
    metrics_novel = Evaluate(model,
                            dataset_novel_val)

    # # aggregation and print results
    # metrics_full = Evaluate(model,
    #                         dataset_full_test)
    
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