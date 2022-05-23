import os
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import torch
import wandb

BASE_DIR = 'punct-data/'

dataset = {
    
}

for filename in os.listdir(BASE_DIR):
  dataset[filename.replace('.csv', '')] = pd.read_csv(os.path.join(BASE_DIR, filename)).dropna()


wandb.login()



# Create a new run
with wandb.init(project="bert-base-cased-ner") as run:

  # Connect an Artifact to the run
  model_name = "neuralmind/bert-base-portuguese-cased"
  #my_model_artifact = run.use_artifact(my_model_name)

  # Download model weights to a folder and return the path
  #model_dir = my_model_artifact.download()
  train_args = {
      'evaluate_during_training': True,
      'logging_steps': 10,
      'num_train_epochs': 15,
      'evaluate_during_training_steps': 5,
      'save_eval_checkpoints': False,
      'train_batch_size': 16,
      'eval_batch_size': 8,
      'overwrite_output_dir': True,
      'labels_list':dataset['train'].labels.unique().tolist(),
      'use_early_stopping':True
  }


  model = NERModel(
      "bert",
      model_name,
      args=train_args,
      use_cuda=torch.cuda.is_available()
  )
  model.train_model(dataset['train'], eval_data=dataset['val'])
  result, model_outputs, wrong_preds = model.eval_model(dataset['val'])

  pd.DataFrame.from_dict(result, orient='index').T.to_csv('test_result.csv', index=False,index_label=False)
