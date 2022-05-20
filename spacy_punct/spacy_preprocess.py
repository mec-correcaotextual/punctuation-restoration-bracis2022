from collections import defaultdict
from pathlib import Path
from spacy.matcher import Matcher
from spacy.tokens import Span
from tqdm import tqdm
import spacy
import re
import pandas as pd
import pickle
import argparse

parser = argparse.ArgumentParser(description='Process CONLL data.')

parser.add_argument('--output_file', type=str,
                    help='output filename')


parser.add_argument('--input_file', type=str,
                    help='It must be a dataframe with headers sentence_id,words,label') 


parser.add_argument('--spacy_model', type=str, default=None,
                    help='It must one of such models: pt_core_news_sm, pt_core_news_md, pt_core_news_lg') 

args = parser.parse_args()


if args.spacy_model in ['pt_core_news_sm', 'pt_core_news_md', 'pt_core_news_lg']:
    nlp = spacy.load(args.spacy_model)
else:
    nlp = spacy.blank('pt')


dataframe = pd.read_csv(args.input_file)
label2dict = {"COMMA":",","PERIOD": ".", "QUESTION":"?"}
TOTAL = len(list(dataframe.groupby("sentence_id")))

TRAIN_DATA = []
for id, group in tqdm(dataframe.groupby("sentence_id"), total=TOTAL):
  text = " ".join(group.words)
  doc = nlp.make_doc(text)
  ents = []

  
  count_dict = defaultdict(lambda : 0)
  
  for i, label in enumerate(group.labels):
     
      if label != "O":
      
        matching_word = group.words.tolist()[i]
       
        span = Span(doc, i, i+1, label=label)
        
        ents.append((span.start_char, span.end_char, label.replace('I-','')))
      
        count_dict[matching_word] += 1
        
  ent = (text, {
        'entities': ents
    })
  TRAIN_DATA.append(ent)


with open(args.output_file+'.pkl', 'wb') as handle:
    pickle.dump(TRAIN_DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)



output_dir = 'output_model'
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)