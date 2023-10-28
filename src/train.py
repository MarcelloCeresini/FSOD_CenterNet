DEBUG_EVALUATE = False


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

    config = load_settings(args.settings)

    if os.path.exists(config['training']['save_base_weights_dir']) \
                  and config["training"]["train_base"] \
                  and config["training"]["no_overwrite"]:
        raise ValueError("Cannot overwrite weights")

    os.makedirs(config['training']['save_training_info_dir'], exist_ok=True)

    debug_mode = config['debug']['debug_mode_active']
    device = config['device']

    K = config['data']['K']
    val_K = config['data']['val_K']
    test_K = config['data']['test_K']
    n_repeats_novel_train = config['training']['repeat_novel_training']

    if isinstance(K, int): K = [K]
    if isinstance(val_K, int): val_K = [val_K]
    if isinstance(test_K, int): test_K = [test_K]

    # Dataset generator. Only one of these has to be instantiated. It always returns
    dataset_gen = DatasetsGenerator(config)
    
    if config['training']['train_base']:

        with wandb.init(project="FSOD_CenterNet", 
                        group="base_training",
                        entity="marcello-e-federico",
                        config=config):

            # Instantiate the model (only the base part)
            model = Model(config, n_base_classes=len(dataset_gen.train_base.cats))
            model = model.to(device)

            wandb.watch(model, log='all', log_freq=config['debug']['wandb_watch_model_freq'])

            # Use the dataset generator to generate the base set
            dataset_base_train, dataset_base_val, dataset_base_test = dataset_gen.get_base_sets_dataloaders(
                config['training']['batch_size'], config['training']['num_workers'],
                config['training']['pin_memory'], config['training']['drop_last'], 
                shuffle=True
            )

            if debug_mode:
                print("Dataset base train length: ", len(dataset_base_train))
            #     # sample, landmarks, original_image_size = dataset_base_train[0]
            #     # use "show_images.py" functions to show the sample / samples

            optimizer_base = Adam([
                    {'params': model.encoder.parameters(), 'lr': config['training']['base']['encoder_lr']},
                    {'params': [t for k, t in model.named_parameters() if 'encoder' not in k]}
                ], 
                lr=config['training']['base']['lr'],
                weight_decay=config['training']['base']['weight_decay'])

            scheduler_base = T.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_base, patience=config['training']['base']['reduce_lr_patience']
            )
        
            # Train the base model and save final weights
            if not DEBUG_EVALUATE:
                model = train_loop( model,
                                    config,
                                    training_loader=dataset_base_train,
                                    validation_loader=dataset_base_val,
                                    optimizer=optimizer_base,
                                    scheduler=scheduler_base,
                                    device=device)
            
            # Evaluation on base train dataset
            metrics = Evaluate(
                model, 
                dataset_base_train, 
                prefix="train/",
                device=device,
                config=config
            )(is_novel=False)
            # Evaluation also on base test dataset
            metrics_test = Evaluate(
                model, 
                dataset_base_test, 
                prefix="test/",
                device=device,
                config=config
            )(is_novel=False)
            # add to the "metrics" dict also the info over the test set, to dump it together
            metrics.update(metrics_test)

            # TODO: add timestamp so that it doesn't overwrite the same metrics over and over
            with open(os.path.join(config['training']['save_training_info_dir'], 
                                   config['training']['base_stats_save_name']), 'wb') as f:
                pickle.dump(metrics, f)

    ## NOVEL TRAININGS ##
    if config['training']['train_novel']:
        
        metrics_novel_list = {k: [] for k in K}
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
        model = Model(  config=config, 
                        n_base_classes = len(dataset_gen.train_base.cats),
                        n_novel_classes=config['data']['novel_classes_to_sample'])

        # START TRAINING!
        total_trainings = len(K) * n_repeats_novel_train
        for i in tqdm(range(total_trainings), position=0, desc="Novel trainings: ", leave=True):

            with wandb.init(project="FSOD_CenterNet", 
                            group="novel_training",
                            config=config):

                current_train_K = K[i % n_repeats_novel_train]
                current_val_K   = val_K[i % n_repeats_novel_train]
                current_test_K  = test_K[i % n_repeats_novel_train]

                # Reload weights from base training + freeze base part
                model = set_model_to_train_novel(model, config)
                model = model.to(device)

                # Optimizer
                optimizer_novel = Adam(model.parameters(), 
                                       lr=config['training']['novel']['lr'],
                                       weight_decay=config['training']['novel']['weight_decay'])

                scheduler_novel = T.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer_novel, patience=config['training']['novel']['reduce_lr_patience']
                )

                # Obtain dataset from generator
                # Use the dataset generator to generate the base set
                _, (dataset_novel_train, dataset_novel_val, dataset_novel_test) = \
                    dataset_gen.generate_dataloaders(
                        K=current_train_K, val_K=current_val_K, test_K=current_test_K,
                        num_novel_classes_to_sample=config['data']['novel_classes_to_sample'],
                        novel_classes_to_sample_list=config['data']['novel_classes_list'],
                        gen_random_seed=config['data']['gen_random_seed'],
                        batch_size=config['training']['batch_size'],
                        num_workers=config['training']['num_workers'],
                        pin_memory=config['training']['pin_memory'],
                        drop_last=config['training']['drop_last'],
                        shuffle=True
                    )

                model = train_loop(model,
                    config=config,
                    training_loader=dataset_novel_train,
                    validation_loader=dataset_novel_val,
                    optimizer=optimizer_novel,
                    scheduler=scheduler_novel,
                    device=device,
                    novel_training=True)
                
                wandb.watch(model, log='all', log_freq=config['debug']['wandb_watch_model_freq'])

                # Evaluation on novel_dataset
                metrics_novel = Evaluate(model, dataset_novel_train, device, config, prefix="train/")(is_novel=True)
                m = Evaluate(model, dataset_novel_test, device, config, prefix="test/")(is_novel=True)
                metrics_novel.update(m)
                metrics_novel_list[current_train_K].append(metrics_novel)

                # TODO: merge the two datasets
                # TODO: can we do this in another file? It may happen that we only train on base or on novel
                # ANSWER: we don't save the weights of the novel heads, so we can't evaluate after this point (or otherwise we save the weights)
                # Evaluation on full dataset
                # metrics_full = Evaluate(model, dataset_full_test)
                # metrics_full_list[current_train_K].append(metrics_full)

            with open(os.path.join(config['training']['save_training_info_dir'], 
                                   config['training']['novel_stats_save_name']), 'wb') as f:
                pickle.dump(metrics_novel_list, f)

            # with open(os.path.join(conf['training']['save_training_info_dir'], 'full_training_info.pkl'), 'wb') as f:
            #     pickle.dump(metrics_full_list, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)