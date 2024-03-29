# -*- coding: utf-8 -*-
"""BertPunctuator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/191TmDhyzcrfzr77G3_-xe2zxcQVR8j8_
"""

import os
import pandas as pd
import torch
import wandb
from simpletransformers.ner import NERModel, NERArgs
import argparse
import shutil
from evaluate import evaluate
from preprocess import preprocess

parser = argparse.ArgumentParser(description='Process dataframe data.')

parser.add_argument('--path_to_data',
                    default='./data/',
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--dataset',
                    default='tedtalk2012',
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--n_epochs',
                    default=12,
                    type=int,
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--k_fold_eval',
                    action='store_true',
                    default=False,
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--bert_model', default="neuralmind/bert-base-portuguese-cased",
                    help='It must one of such models valid bert model, see hugginface plataform.')

args = parser.parse_args()

DATASET_NAME = os.path.split(args.path_to_data)[-1]
BASE_DIR = f'../texts/{args.dataset}/'

wandb.login(key='8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4')

if args.k_fold_eval:
    print('\nRunning k-fold evaluation...')
    results_ents, results_micro_avg = [], []
    for folder in sorted(os.listdir(BASE_DIR)):

        if os.path.isdir(os.path.join(BASE_DIR, folder)):
            print(f'\nRunning on {folder}')
            dataset_path = os.path.join(BASE_DIR, folder)
            out_path = os.path.join(args.path_to_data, folder)
            os.makedirs(out_path, exist_ok=True)
            preprocess(dataset_path, out_path)  # preprocess dataset

            dataset = {filename.replace('.csv', ''): pd.read_csv(os.path.join(out_path, filename)).dropna()
                       for filename in os.listdir(out_path)}

            # Create a new run
            project = "punctuation-restoration-kfold"

            # Download model weights to a folder and return the path
            # model_dir = my_model_artifact.download()
            train_args = {
                "silent": None,
                'evaluate_during_training': True,
                'logging_steps': 10,
                'num_train_epochs': args.n_epochs,
                'evaluate_during_training_steps': dataset['train'].shape[0],
                'train_batch_size': 32,
                'eval_batch_size': 8,
                'overwrite_output_dir': True,
                'save_eval_checkpoints': False,
                'save_model_every_epoch': False,
                'save_steps': -1,
                'labels_list': ["O", "I-COMMA", "I-PERIOD", "I-QUESTION"],
                'use_early_stopping': True,
                'wandb_project': project,
                'wandb_kwargs': {'name': 'bert-base-' + folder},
            }
            print("\nCleaning up previous runs...")
            shutil.rmtree('./outputs/', ignore_errors=True)
            # # Create a new NERModel
            print("\nTraining model...")
            model = NERModel(
                "bert",
                args.bert_model,
                args=train_args,
                use_cuda=torch.cuda.is_available()
            )
            model.train_model(dataset['train'], eval_data=dataset['dev'])
            print("\nEvaluation model...")
            # Evaluate the model
            model_dir = './outputs/best_model/'

            eval_args = NERArgs()
            eval_args.labels_list = ["O", "COMMA", "PERIOD", "QUESTION"]
            model = NERModel(
                "bert",
                model_dir,
                args=eval_args,
                use_cuda=torch.cuda.is_available()
            )
            micro_avg, ents = evaluate(model, dataset['test'])
            micro_avg.update({'dataset_name': folder, 'classifier_name': 'bert-base'})
            results_micro_avg.append(micro_avg)

            ents.update({'dataset_name': folder, 'classifier_name': 'bert-base'})
            results_ents.append(pd.DataFrame(ents))
            # saves the model
            artifact = wandb.Artifact('bert-model', type='model')
            artifact.add_dir(model_dir)

    os.makedirs('./results/', exist_ok=True)
    pd.DataFrame(results_micro_avg).to_csv('./results/micro_avg_results.csv')
    pd.concat(results_ents).to_csv('./results/micro_avg_ents_results.csv')

else:

    preprocess(BASE_DIR, args.path_to_data)

    dataset = {filename.replace('.csv', ''): pd.read_csv(os.path.join(args.path_to_data, filename)).dropna()
               for filename in os.listdir(args.path_to_data)
               if os.path.isfile(os.path.join(args.path_to_data, filename))}

    # Create a new run
    project = "punctuation-restoration"
    # Connect an Artifact to the run

    # Download model weights to a folder and return the path
    # model_dir = my_model_artifact.download()
    train_args = {
        'evaluate_during_training': True,
        'logging_steps': 10,
        'num_train_epochs': 12,
        'evaluate_during_training_steps': dataset['train'].shape[0],
        'train_batch_size': 32,
        'eval_batch_size': 8,
        'overwrite_output_dir': True,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': -1,
        'labels_list': dataset['train'].labels.unique().tolist(),
        'use_early_stopping': True,
        'wandb_project': project,
        'wandb_kwargs': {'name': 'bert-base-all'},
    }

    model = NERModel(
        "bert",
        args.bert_model,
        args=train_args,
        use_cuda=torch.cuda.is_available()
    )
    model.train_model(dataset['train'], eval_data=dataset['dev'])
    model_name = './outputs/best_model/'
    model = NERModel(
        "bert",
        model_name,
        args=train_args,
        use_cuda=torch.cuda.is_available()
    )
    micro_avg, ents = evaluate(model, dataset['test'])
    pd.DataFrame.from_dict(micro_avg).to_csv('./outputs/best_model/micro_avg_results.csv')
    pd.DataFrame.from_dict(ents).to_csv('./outputs/best_model/micro_avg_results.csv')
