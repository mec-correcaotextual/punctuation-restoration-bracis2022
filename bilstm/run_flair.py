import os
import wandb
from flair.data import Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import SGDW
from utils import generate_test_file
import argparse

parser = argparse.ArgumentParser(description='Process dataframe data.')

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

    parser.add_argument('--use_crf', default=True, action='store_true')

    args = parser.parse_args()
    # corpus_name = 'tiago_tedtalk2012'
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
        bert_embedding = TransformerWordEmbeddings('neuralmind/bert-base-portuguese-cased', layers='-1', layer_mean=False)
        embedding_types.append(bert_embedding)
        model_dir += f'_{embedding_name}'
        sentence = Sentence('The grass is green.')
        embeddings.embed(sentence)
        print(f'Embedding size: {sentence[0].embedding.size()}')

    embeddings = StackedEmbeddings(embeddings=embedding_types)
    if args.use_crf:
        model_dir += '_crf'
        print('\nRunning using CRF')

    model_dir = os.path.join(model_dir, corpus_name)

    os.makedirs(model_dir, exist_ok=True)
    columns = {0: 'token', 1: 'ner'}
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

    n_epochs = 100
    batch_size = 32
    project = "punctuation-restoration"
    with wandb.init(project=project) as run:

        run.name = f'bilstm_{embedding_name}'
        trainer.train(model_dir, optimizer=SGDW, learning_rate=0.1, mini_batch_size=batch_size, max_epochs=n_epochs)

    test_results_file = os.path.join(model_dir, 'test.tsv')
    new_test_file = os.path.join(model_dir, corpus_name + '_conlleval_test.tsv')
    test_results = generate_test_file(test_results_file, new_test_file)
