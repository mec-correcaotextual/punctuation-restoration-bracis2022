import os
import numpy as np

from sklearn_crfsuite import CRF
from seqeval.metrics import classification_report

from preprocess import preprocess
from utils import read_corpus_file, data_preprocessing, convert_data, dump_report
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Process dataframe data.')

BASE_DIR = '../texts/tedtalk2012/'


def run(args):
    # corpus_name = 'obras'
    corpus_name = 'tedtalk2012'

    report_dir = f'./{corpus_name}'

    preprocess(BASE_DIR, args.path_to_data)
    train_file = os.path.join(args.path_to_data, 'train.csv')
    test_file = os.path.join(args.path_to_data, 'test.csv')

    report_file = os.path.join(report_dir, corpus_name + '_crf.csv')

    test_data = read_corpus_file(test_file, split_char=',')
    train_data = read_corpus_file(train_file, split_char=',')

    test_data_original = np.array(test_data, dtype=object)

    print('\n  Train data:', len(train_data))
    print('  Test data:', len(test_data))

    print('\nPreprocessing ...')

    print('\n  Train data')

    train_data = data_preprocessing(train_data)

    print('  Test data')

    test_data = data_preprocessing(test_data)

    X_train, y_train = convert_data(train_data)
    X_test, y_test = convert_data(test_data)
    pd.DataFrame.from_dict(X_train[0]).T.to_csv(f'{corpus_name}_X_train.csv', index=False)
    print('\nExample features:', X_train[0])
    print('Tags:', y_train[0])

    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)

    print('\nEvaluating CRF')

    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)

    dict_report = classification_report(y_test, y_pred, output_dict=True)

    data_conll = ''

    for data, real_tags, pred_tags in zip(test_data, y_test, y_pred):
        words = data[0]
        sent = '\n'.join('{0} {1} {2}'.format(word, real_tag, pred_tag)
                         for word, real_tag, pred_tag in zip(words, real_tags, pred_tags))
        sent += '\n\n'
        data_conll += sent

    print('\nReport:', dict_report)

    print('\nSaving the report in:', report_file)

    dump_report(dict_report, report_file)

    script_result_file = os.path.join(report_dir, corpus_name + '_crf.tsv')

    with open(script_result_file, 'w', encoding='utf-8') as file:
        file.write(data_conll)


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

    parser.add_argument('--k_fold_eval',
                        action='store_true',
                        default=False,
                        help='Files must be a dataframe with headers sentence_id,words,label')

    args = parser.parse_args()
    run(args)
