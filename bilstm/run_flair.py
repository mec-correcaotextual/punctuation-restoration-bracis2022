import os
import wandb

from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.optim import SGDW
from utils import generate_test_file
from transformers import TrainingArguments, Trainer

if __name__ == '__main__':

    # corpus_name = 'tiago_tedtalk2012'
    corpus_name = 'tedtalk2012'
    glove_file = './embeddings/glove_s300.gensim'
    word2vec_cbow_file = './embeddings/cbow_s300.gensim'
    word2vec_skip_file = './embeddings/skip_s300.gensim'

    is_use_glove = False
    is_use_w2v_skip = True
    is_use_w2v_cbow = False

    is_use_crf = True

    columns = None
    data_folder = None

    n_epochs = 100

    batch_size = 32

    train_file = None
    test_file = None
    val_file = None

    if corpus_name == 'tiago_obras':
        columns = {0: 'token', 1: 'ner'}
        data_folder = './data/obras'
        train_file = 'train.csv'
        val_file = 'dev.csv'
        test_file = 'test.csv'
    elif corpus_name == 'tedtalk2012':
        columns = {0: 'token', 1: 'ner'}
        data_folder = './data/tedtalk2012'
        train_file = 'train.csv'
        val_file = 'dev.csv'
        test_file = 'test.csv'
    else:
        print('Corpus option invalid!')
        exit(0)

    model_dir = './models/bilstm'

    if is_use_w2v_skip:
        model_dir += '_w2v_skip'
    elif is_use_w2v_cbow:
        model_dir += '_w2v_cbow'
    elif is_use_glove:
        model_dir += '_glove'

    if is_use_crf:
        model_dir += '_crf'
        print('\nRunning using CRF')

    print('\n')

    model_dir = os.path.join(model_dir, corpus_name)

    os.makedirs(model_dir, exist_ok=True)

    corpus = ColumnCorpus(data_folder, columns, train_file=train_file, test_file=test_file, dev_file=val_file)

    print('\nTrain len: ', len(corpus.train))
    print('Dev len: ', len(corpus.dev))
    print('Test len: ', len(corpus.test))

    print('\nTrain: ', corpus.train[0].to_tagged_string('label'))
    print('Dev: ', corpus.dev[0].to_tagged_string('label'))
    print('Test: ', corpus.test[0].to_tagged_string('label'))

    tag_type = 'ner'

    tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)

    print('\nTags: ', tag_dictionary.idx2item)

    # Loading Traditional Embeddings

    traditional_embedding = None

    if is_use_w2v_skip:
        print('\nRunning using Word2vec Skip')
        traditional_embedding = WordEmbeddings(word2vec_skip_file)
    if is_use_w2v_cbow:
        print('\nRunning using Word2vec CBOW')
        traditional_embedding = WordEmbeddings(word2vec_cbow_file)
    elif is_use_glove:
        print('\nRunning using Glove')
        traditional_embedding = WordEmbeddings(glove_file)
    else:
        print('\nNot using Traditional embedding')

    # Loading Contextual Embeddings

    embedding_types = []

    if traditional_embedding is not None:
        embedding_types.append(traditional_embedding)

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256, embeddings=embeddings, tag_dictionary=tag_dictionary,
                            tag_type=tag_type, use_crf=is_use_crf)
    wandb.login(key='8e593ae9d0788bae2e0a84d07de0e76f5cf3dcf4')

    with wandb.init(project="bert-base-punct", entity="tblima") as run:
        run.name = f'bilstm_{corpus_name}'
        args = TrainingArguments(output_dir=model_dir, learning_rate=0.1, max_steps=n_epochs,
                                 per_device_train_batch_size=batch_size, report_to="wandb")
        trainer = ModelTrainer(tagger, corpus, args=args)

        trainer.train(model_dir, optimizer=SGDW)

    test_results_file = os.path.join(model_dir, 'test.tsv')

    new_test_file = os.path.join(model_dir, corpus_name + '_conlleval_test.tsv')

    test_results = generate_test_file(test_results_file, new_test_file)
