import argparse
import os
from collections import defaultdict
import pandas as pd
import spacy
import torch
from simpletransformers.ner import NERModel, NERArgs
from spacy.scorer import get_ner_prf
from spacy.training import Example
from tqdm import tqdm
from spacy.tokens import Span

nlp = spacy.blank('pt')


def preprocess_data(dataframe):
    TOTAL = len(list(dataframe.groupby("sentence_id")))
    data = []
    for _, group in tqdm(dataframe.groupby("sentence_id"), total=TOTAL):
        text = " ".join(group.words)
        doc = nlp.make_doc(text)
        ents = []
        count_dict = defaultdict(lambda: 0)
        for i, label in enumerate(group.labels):
            if label != "O":
                matching_word = group.words.tolist()[i]
                span = Span(doc, i, i + 1, label=label)
                ents.append((span.start_char, span.end_char, label.replace('I-', '')))
                count_dict[matching_word] += 1
        ent = (text, {
            'entities': ents
        })
        data.append(ent)
    return data


parser = argparse.ArgumentParser(description='Process dataframe data.')

parser.add_argument('--test_df',
                    help='input files', default='./data/tedtalk2012/test.csv')

parser.add_argument('--iters',
                    help='Number of tests', default=10, type=int)

parser.add_argument('--result_path',
                    help='results path', default='./results/', type=str)

parser.add_argument('--bert_model', default="./outputs/",
                    help='It must one of such models valid bertpunctuator model, see hugginface plataform or dir.')
args = parser.parse_args()


def evaluate(model, dataset):
    TEST_DATA = preprocess_data(dataset)

    model_args = NERArgs()
    model_args.labels_list = ["O", "COMMA", "PERIOD", "QUESTION"]

    y_true = []
    texts = []
    for _, group in dataset.groupby("sentence_id"):
        text = " ".join(group.words)
        texts.append(text)
        y_true.append(group.labels.apply(lambda label: label.replace("I-", "")).tolist())

    predictions = model.predict(texts)

    y_pred = []
    for i, pred in enumerate(predictions[0], 1):
        y_pred.append(list(map(lambda item: list(item.values())[0].replace("I-", ""), pred)))

    predictions_ner = predictions[0]

    examples = []
    for i, (text, entities) in enumerate(TEST_DATA):
        doc = nlp(text)
        doc.set_ents([Span(doc, i, i + 1, list(item.values())[0])
                      for i, item in enumerate(predictions_ner[i])
                      if list(item.values())[0] != "O"])

        example = Example.from_dict(doc, entities)
        examples.append(example)

    scores = get_ner_prf(examples)

    micro_avg = {
        'f1-score': scores.pop('ents_f'),
        'pprecision': scores.pop('ents_p'),
        'recall': scores.pop('ents_r')
    }

    ents_per_type = scores.pop('ents_per_type')

    return micro_avg, ents_per_type


if __name__ == '__main__':
    model_args = NERArgs()
    model_args.labels_list = ["O", "COMMA", "PERIOD", "QUESTION"]
    evaluate(NERModel(args.bert_model, model_args), pd.read_csv(args.test_df))
