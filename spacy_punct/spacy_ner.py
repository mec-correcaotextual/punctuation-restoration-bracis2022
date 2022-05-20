from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm
from collections import Counter
import pandas as pd
import re 
from collections import defaultdict
from spacy.matcher import Matcher
from spacy.tokens import Span
from tqdm import tqdm
import spacy
import pickle
import argparse
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer


a = {'hello': 'world'}


parser = argparse.ArgumentParser(description='Process CONLL data.')

parser.add_argument('--output_file', type=str,
                    help='output filename')


parser.add_argument('--input_files', type=str,
                    help='It must be pickle files of train, dev, test separated by commas' + 
                            'ex.: train,dev,test') 

parser.add_argument('--iters', type=int,
                    help='Number of interation for training') 

parser.add_argument('--input_files', type=int,default=32,
                    help='It must be pickle files of train.pkl, dev.pkl, test.pkl separated by commas' + 
                            'ex.: train,dev,test') 

parser.add_argument('--spacy_model', type=str, default=None,
                    help='It must be a dataframe with headers sentence_id,words,label') 

args = parser.parse_args()


if args.spacy_model in ['pt_core_news_sm', 'pt_core_news_md', 'pt_core_news_lg']:
    nlp = spacy.load(args.spacy_model)
else:
    nlp = spacy.blank('pt')

files_dict = {}.fromkeys(['train','dev','test'])
for input_file in args.input_files.split(","):
  files_dict[input_file.split(".")[0]] = pd.read_pickle(input_file)


def train_spacy(data, iterations=10, batch_size=32, model=None):
   
    if model is not None:
      nlp = spacy.load(model)
      print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('pt')  
        print("Created blank 'pt' model")

    #set up the pipeline

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Init loss
    losses = []

    # Init and configure optimizer
    optimizer = nlp.create_optimizer()
    optimizer.learn_rate = 0.001  # Change to some lr you prefers
    batch_size = 32  # Choose batch size you prefers

    for itn in range(iterations):
        print("Starting iterations " + str(itn))
        random.shuffle(data)
        loss = {}

        # Batch the examples and iterate over them
        for batch in spacy.util.minibatch(data, size=batch_size):
            # Create Example instance for each training example in mini batch
            #print(annotations)
            examples = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in batch]
            # Update model with mini batch
            
            loss = nlp.update(examples, drop=0.2, sgd=optimizer)
            losses.append(loss)

    loss_df = pd.DataFrame.from_dict(losses,orient='index').T
    loss_df.plot()
    loss_df.to_csv('loss.csv')
    
    return nlp


train_spacy(files_dict['train'], iterations=args.iters, batch_size=args.batch_size, model=args.spacy_model)


dev_examples = []
for text, entities in files_dict['dev']:
    doc = nlp(text)  
    example = Example.from_dict(doc, entities)
    dev_examples.append(example)

scorer = Scorer(nlp)
scores = Scorer.score_spans(dev_examples, "ents")
print('Test scores: ', scores['ents_f'] * 100)


test_examples = []
for text, entities in files_dict['test']:
    doc = nlp(text)  
    example = Example.from_dict(doc, entities)
    test_examples.append(example)

scorer = Scorer(nlp)
scores = Scorer.score_spans(test_examples, "ents")
print('Test scores: ', scores['ents_f'] * 100)
pd.DataFrame(scores).to_csv('test_scores.csv')







