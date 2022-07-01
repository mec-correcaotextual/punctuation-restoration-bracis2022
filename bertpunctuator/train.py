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
from simpletransformers.ner import NERModel
import argparse

parser = argparse.ArgumentParser(description='Process dataframe data.')

parser.add_argument('--path_to_data',
                    default='./data/',
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--dataset',
                    default='tedtalk2012',
                    help='Files must be a dataframe with headers sentence_id,words,label')

parser.add_argument('--bert_model', default="neuralmind/bert-base-portuguese-cased",
                    help='It must one of such models valid bert model, see hugginface plataform.')

args = parser.parse_args()

DATASET_NAME = os.path.split(args.path_to_data)[-1]
BASE_DIR = os.path.join(args.path_to_data, args.dataset)

dataset = {filename.replace('.csv', ''): pd.read_csv(os.path.join(BASE_DIR, filename)).dropna()
           for filename in os.listdir(BASE_DIR)}

wandb.login(key='8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4')

# Create a new run
project = "punctuation-restoration"
# Connect an Artifact to the run
model_name = args.bert_model

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
    'wandb_kwargs': {'name': 'bert-base'},
}

model = NERModel(
    "bert",
    model_name,
    args=train_args,
    use_cuda=torch.cuda.is_available()
)
model.train_model(dataset['train'], eval_data=dataset['dev'])
result, model_outputs, wrong_preds = model.eval_model(dataset['test'])

if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

pd.DataFrame.from_dict(result, orient='index').T.to_csv(os.path.join(args.result_path, 'overall_model_result.csv'),
                                                        index=False, index_label=False)
