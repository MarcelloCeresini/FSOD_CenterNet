# import from third parties
import torch as T
from torch.optim import Adam

import os

# import from own packages
from model import Model
from training import train_loop_base, set_model_to_train_novel
from data_pipeline import TransformTraining, TransformTesting, get_data_loaders
from evaluation import Evaluate
from config import Config

debugging = True


if __name__ == "__main__":

    conf = Config()

    assert (not os.path.exists(conf.weights_path)), "weights path already exists, will not overwrite so delete it first or change the path " + \
        "current path: {}".format(conf.weights_path)

    model = Model(encoder_name="resnet18",  # TODO: maybe the following in config?
                n_base_classes=100,
                n_novel_classes=10,
                head_base_heatmap_mode="CosHead",
                head_novel_heatmap_mode="AdaptiveCosHead")


    # base dataset
    # dataset_base_train, \
    #     dataset_novel_train = get_data_loaders(annotations_path=["base_train.json", # TODO: change this path (maybe put in config?)
    #                                                              "novel_train.json"], # TODO: change this path
    #                                            images_dir="always_the_same_dir", # TODO: change this path
    #                                            transform=TransformTraining(),
    #                                            batch_size=conf.train_batch_size,
    #                                            num_workers=conf.num_workers,
    #                                            pin_memory=conf.pin_memory,
    #                                            drop_last=conf.drop_last,
    #                                            shuffle=True)

    # dataset_base_test, dataset_novel_test, \
    #     dataset_full_test = get_data_loaders(annotations_path=["base_test.json", # TODO: change this path
    #                                                            "novel_test.json", # TODO: change this path
    #                                                            "full_test.json"], # TODO: change this path
    #                                          images_dir="always_the_same_dir", # TODO: change this path
    #                                          transform=TransformTesting(),
    #                                          batch_size=conf.train_batch_size,
    #                                          num_workers=conf.num_workers,
    #                                          pin_memory=conf.pin_memory,
    #                                          drop_last=conf.drop_last,
    #                                          shuffle=False)


    if debugging:
        print("Dataset base train length: ", len(dataset_base_train))
        sample, landmarks, original_image_size = dataset_base_train[0]
        # use "show_images.py" functions to show the sample / samples

    optimizer_base = Adam(model.parameters(),
                          lr=conf.lr_base)
    
    weights_path = train_loop_base(model,
                                   epochs=conf.epochs_base,
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
    metrics_base = Evaluate(model,
                            dataset_base_test)
    
    # train and eval novel
    metrics_novel_list = []
    for i, (dataset_novel_train, dataset_novel_val, dataset_novel_test) in enumerate(dataset_novel_list):
        print(f"Training on novel dataset n°{i} out of {len(dataset_novel_list)}")
        model.load_state_dict(T.load(weights_path))

        # freeze the weights of everything except the novel head
        model = set_model_to_train_novel(model)


    

        # evaluation on novel_dataset
        metrics_novel = Evaluate(model,
                                dataset_novel_test)

        # aggregation and print results
        metrics_full = Evaluate(model,
                                dataset_full_test)
    
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
        - classes (:class:`~torch.Tensor`), list of all observed classes'''