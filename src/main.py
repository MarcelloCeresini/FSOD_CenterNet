# import from third parties
import torch as T
from torch.optim import Adam

import os, sys

# import from own packages
from model import Model
from training import train_loop, set_model_to_train_novel
from data_pipeline import TransformTraining, TransformTesting, get_data_loaders
from evaluation import Evaluate
from config import Config

debugging = True


if __name__ == "__main__":

    conf = Config()

    assert (not os.path.exists(conf.weights_path)), "weights path already exists, will not overwrite so delete it first or change the path " + \
        "current path: {}".format(conf.weights_path)

    model = Model(encoder_name="resnet18",  # TODO: maybe the following in config?
                n_base_classes=800,
                n_novel_classes=10,
                head_base_heatmap_mode="CosHead",
                head_novel_heatmap_mode="AdaptiveCosHead")

    # TODO: this is only for testing during pre-testing phase
    # base dataset
    dataset_base_train = get_data_loaders(annotations_path=conf.path_train_short, # TODO: change this path
                                               images_dir="always_the_same_dir", # TODO: change this path
                                               transform=TransformTraining(),
                                               batch_size=conf.train_batch_size,
                                               num_workers=conf.num_workers,
                                               pin_memory=conf.pin_memory,
                                               drop_last=conf.drop_last,
                                               shuffle=True)
    
    dataset_base_val = get_data_loaders(annotations_path=conf.path_train_short, # TODO: change this path
                                               images_dir="always_the_same_dir", # TODO: change this path
                                               transform=TransformTraining(),
                                               batch_size=conf.train_batch_size,
                                               num_workers=conf.num_workers,
                                               pin_memory=conf.pin_memory,
                                               drop_last=conf.drop_last,
                                               shuffle=True)
    
    dataset_base_test = get_data_loaders(annotations_path=conf.path_train_short, # TODO: change this path
                                               images_dir="always_the_same_dir", # TODO: change this path
                                               transform=TransformTesting(),
                                               batch_size=conf.train_batch_size,
                                               num_workers=conf.num_workers,
                                               pin_memory=conf.pin_memory,
                                               drop_last=conf.drop_last,
                                               shuffle=True)
    
    dataset_novel_list = []

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
    
    model = train_loop(model,
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
    # TODO: check that empty novel classes don't affect the evaluation
    metrics_base = Evaluate(model,
                            dataset_base_test)
    
    # TODO: save them as a json
    print(metrics_base)
    
    # train and eval novel
    metrics_novel_list = []
    metrics_full_list = []
    for i, (dataset_novel_train, dataset_novel_val, dataset_novel_test) in enumerate(dataset_novel_list):
        print(f"Training on novel dataset n°{i} out of {len(dataset_novel_list)}")
        
        model = set_model_to_train_novel(model, conf)

        optimizer_novel = Adam(model.parameters(),
                          lr=conf.lr_novel)

        model = train_loop(model,
                           epochs=conf.epochs_novel,
                           training_loader=dataset_novel_train,
                           validation_loader=dataset_novel_val,
                           optimizer=optimizer_novel,
                           novel_training=True)

        # evaluation on novel_dataset
        metrics_novel_list.append(Evaluate(model,
                                           dataset_novel_test))

        # TODO: how to combine base and novel datasets?

        # aggregation and print results
        metrics_full_list.append(Evaluate(model,
                                          dataset_full_test))
    
    