import os
import shutil

import wandb
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import SGDW
import pandas as pd

from evaluate import evaluate
from preprocess import preprocess
from utils import generate_test_file
import argparse

parser = argparse.ArgumentParser(description='Process dataframe data.')


def train(args):
    corpus_name = args.dataset

    model_dir = './models/bilstm'

    embeddings = None
    embedding_name = args.embeddings
    embedding_types = []
    if embedding_name == 'skip_s300':

        print(f'\nRunning using {args.embeddings}')
        traditional_embedding = WordEmbeddings('./embeddings/skip_s300.gensim')

        if traditional_embedding is not None:
            embedding_types.append(traditional_embedding)

        embedding_name = args.embeddings.split('/')[-1].split('.')[0]
        model_dir += f'_{embedding_name}'

    elif embedding_name == 'bert':
        bert_embedding = TransformerWordEmbeddings('neuralmind/bert-base-portuguese-cased', layers='-1',
                                                   layer_mean=False)
        embedding_types.append(bert_embedding)
        model_dir += f'_{embedding_name}'
        sentence = Sentence('The grass is green.')
        bert_embedding.embed(sentence)
        print(f'Embedding size: {sentence[0].embedding.size()}')

    embeddings = StackedEmbeddings(embeddings=embedding_types)
    if args.use_crf:
        model_dir += '_crf'
        print('\nRunning using CRF')

    model_dir = os.path.join(model_dir, corpus_name)

    os.makedirs(model_dir, exist_ok=True)
    columns = {0: 'token', 1: 'ner'}

    BASE_DIR = f'../texts/{args.dataset}/'

    if args.k_fold_eval:
        print('\nRunning k-fold evaluation...')
        results_ents, results_micro_avg = [], []
        for folder in os.listdir(BASE_DIR):

            if os.path.isdir(os.path.join(BASE_DIR, folder)):
                print(f'\nRunning on {folder}')
                dataset_path = os.path.join(BASE_DIR, folder)
                out_path = os.path.join(args.path_to_data, folder)

                print('\nCleaning up previous runs...')
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(out_path, exist_ok=True)

                print(f'\nPreprocessing {dataset_path}')
                preprocess(dataset_path, out_path)  # preprocess dataset

                corpus = ColumnCorpus(out_path, columns)

                # Create a new run
                project = "punctuation-restoration-kfold"

                tag_type = 'ner'

                tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
                tag_dictionary.remove_item('<unk>')
                print('\nTags: ', tag_dictionary.idx2item)

                tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                        tag_type=tag_type, use_crf=args.use_crf)

                trainer = ModelTrainer(tagger, corpus)

                wandb.login(key='8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4')


                batch_size = 32

                with wandb.init(project=project) as run:
                    run.name = f'bilstm_{embedding_name}'
                    trainer.train(model_dir, optimizer=SGDW, learning_rate=0.1, mini_batch_size=batch_size,
                                  max_epochs=args.n_epochs)

                test_results_file = os.path.join(model_dir, 'test.tsv')
                new_test_file = os.path.join(model_dir, corpus_name + '_conlleval_test.tsv')
                generate_test_file(test_results_file, new_test_file)
                micro_avg, per_ents = evaluate(corpus, os.path.join(model_dir, 'best-model.pt'))
                results_micro_avg.append(micro_avg)
                results_ents.append(per_ents)

        os.makedirs('./outputs/', exist_ok=True)
        pd.DataFrame(results_micro_avg).to_csv('./outputs/micro_avg.csv')
        pd.DataFrame(results_ents).to_csv('./outputs/micro_avg_ents.csv')

    else:

        corpus = ColumnCorpus(args.path_to_data, columns)

        print('\nTrain len: ', len(corpus.train))
        print('Dev len: ', len(corpus.dev))
        print('Test len: ', len(corpus.test))

        print('\nTrain: ', corpus.train[0].to_tagged_string('label'))
        print('Dev: ', corpus.dev[0].to_tagged_string('label'))
        print('Test: ', corpus.test[0].to_tagged_string('label'))

        tag_type = 'ner'

        tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
        tag_dictionary.remove_item('<unk>')
        print('\nTags: ', tag_dictionary.idx2item)

        tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                                tag_type=tag_type, use_crf=args.use_crf)

        trainer = ModelTrainer(tagger, corpus)

        wandb.login(key='8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4')


        batch_size = 32
        project = "punctuation-restoration"
        with wandb.init(project=project) as run:

            run.name = f'bilstm_{embedding_name}'
            trainer.train(model_dir, optimizer=SGDW,
                          learning_rate=0.1,
                          mini_batch_size=batch_size,
                          max_epochs=args.n_epochs)

        test_results_file = os.path.join(model_dir, 'test.tsv')
        new_test_file = os.path.join(model_dir, corpus_name + '_conlleval_test.tsv')
        generate_test_file(test_results_file, new_test_file)


if __name__ == '__main__':
    parser.add_argument('--result_path',
                        default='./results/',
                        help='output filename')

    parser.add_argument('--path_to_data',
                        default='./data/tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--dataset',
                        default='tedtalk2012',
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--embeddings', default='skip_s300',
                        help='It must one of such models valid bert model, see hugginface plataform.')

    parser.add_argument('--k_fold_eval',
                        action='store_true',
                        default=False,
                        help='Files must be a dataframe with headers sentence_id,words,label')

    parser.add_argument('--use_crf', default=True, action='store_true')

    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')

    args = parser.parse_args()
    train(args)
