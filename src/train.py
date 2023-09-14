import argparse
import os
import pickle

import torch as T
import yaml
from torch.optim import Adam
from tqdm import tqdm
import wandb

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

    if os.path.exists(conf['training']['save_base_weights_dir']) and conf["train_base"]:
        raise ValueError("Cannot overwrite weights")

    debug_mode = conf['debug']['debug_mode_active']
    device = conf['device']

    K = conf['data']['K']
    val_K = conf['data']['val_K']
    test_K = conf['data']['test_K']
    n_repeats_novel_train = conf['training']['repeat_novel_training']

    if isinstance(K, int): K = [K]
    if isinstance(val_K, int): val_K = [val_K]
    if isinstance(test_K, int): test_K = [test_K]


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
    
    if conf['training']['train_base']:

        with wandb.init(project="FSOD_CenterNet", 
                        group="base_training",
                        config=conf):

            # Instantiate the model (only the base part)
            model = Model(  encoder_name = conf['model']['encoder_name'], 
                            n_base_classes = len(dataset_gen.train_base.cats),
                            head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'])
            model = model.to(device)

            # Use the dataset generator to generate the base set
            dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
                conf['training']['batch_size'], conf['training']['num_workers'],
                conf['training']['pin_memory'], conf['training']['drop_last'], shuffle=True
            )

            if debug_mode:
                print("Dataset base train length: ", len(dataset_base_train))
                # sample, landmarks, original_image_size = dataset_base_train[0]
                # use "show_images.py" functions to show the sample / samples

            optimizer_base = Adam(model.parameters(), 
                                lr=conf['training']['base']['lr'])
        
            # Train the base model and save final weights
            model = train_loop(model,
                            epochs=conf['training']['base']['epochs'],
                            training_loader=dataset_base_train,
                            validation_loader=dataset_base_val,
                            optimizer=optimizer_base,
                            weights_path=conf['training']['save_base_weights_dir'],
                            name="standard_model_base")
            
            # Evaluation on base test dataset
            metrics_base = Evaluate(model, dataset_base_test)
        
            with open(os.path.join(conf['training']['save_training_info_dir'], 'base_training_info.pkl'), 'wb') as f:
                pickle.dump(metrics_base, f)

    ## NOVEL TRAININGS ##

    if conf['training']['train_novel']:
        
        metrics_novel_list = {k: [] for k in conf['data']['K']}
        # TODO: Maybe it's better to evaluate on full in another script
        # metrics_full_list  = {k: [] for k in conf['data']['K']}

        # Check K, val_K and test_K
        if len(K) == 1 and len(K) != n_repeats_novel_train:
            print(f"Found only 1 train K ({K[0]}): assuming it must be used for all the novel trainings")
            K = [K[0]] * n_repeats_novel_train

        if len(val_K) == 1 and len(val_K) != n_repeats_novel_train:
            print(f"Found only 1 val K ({val_K[0]}): assuming it must be used for all the novel trainings")
            val_K = [val_K[0]] * n_repeats_novel_train

        if len(test_K) == 1 and len(test_K) != n_repeats_novel_train:
            print(f"Found only 1 test K ({test_K[0]}): assuming it must be used for all the novel trainings")
            test_K = [test_K[0]] * n_repeats_novel_train

        # Instantiate the model (also the novel part)
        model = Model(  encoder_name = conf['model']['encoder_name'], 
                        n_base_classes = len(dataset_gen.train_base.cats),
                        head_base_heatmap_mode = conf['model']['head_base_heatmap_mode'],
                        n_novel_classes=conf['data']['novel_classes_to_sample'],
                        head_novel_heatmap_mode = conf['model']['head_novel_heatmap_mode'])


        # START TRAINING!
        total_trainings = len(K) * n_repeats_novel_train

        for i in tqdm(range(total_trainings)):

            # TODO: create wandb.config (possibly also stating that it's the "novel" training)
            with wandb.init(project="FSOD_CenterNet", 
                            group="novel_training",
                            config=conf):
                
                print(f"\nTraining on novel dataset: {i}/{total_trainings}")

                current_train_K = K[i % n_repeats_novel_train]
                current_val_K   = val_K[i % n_repeats_novel_train]
                current_test_K  = test_K[i % n_repeats_novel_train]

                # Reload weights from base training + freeze base part
                model = set_model_to_train_novel(model, conf)
                model = model.to(device)

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

                model = train_loop(model,
                                epochs=conf['training']['novel']['epochs'],
                                training_loader=dataset_novel_train,
                                validation_loader=dataset_novel_val,
                                optimizer=optimizer_novel,
                                weights_path=conf['training']['save_novel_weights_dir'],
                                novel_training=True)
                
                # Evaluation on novel_dataset
                metrics_novel = Evaluate(model, dataset_novel_test)
                metrics_novel_list[current_train_K].append(metrics_novel)

                # TODO: merge the two datasets
                # TODO: can we do this in another file? It may happen that we only train on base or on novel
                # ANSWER: we don't save the weights of the novel heads, so we can't evaluate after this point (or otherwise we save the weights)
                # Evaluation on full dataset
                # metrics_full = Evaluate(model, dataset_full_test)
                # metrics_full_list[current_train_K].append(metrics_full)
        
            with open(os.path.join(conf['training']['save_training_info_dir'], 'novel_training_info.pkl'), 'wb') as f:
                pickle.dump(metrics_novel_list, f)
            
            # with open(os.path.join(conf['training']['save_training_info_dir'], 'full_training_info.pkl'), 'wb') as f:
            #     pickle.dump(metrics_full_list, f)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)