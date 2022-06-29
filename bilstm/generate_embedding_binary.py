from gensim.models import KeyedVectors

if __name__ == '__main__':
    embeddings_txt_file = './embeddings/skip_s300.txt'
    embeddings_bin_file = './embeddings/skip_s300.gensim'

    print('\nConverting TXT Embedding to Binary ...')

    emb_vectors = KeyedVectors.load_word2vec_format(embeddings_txt_file, binary=False)

    emb_vectors.save(embeddings_bin_file)

    print('\nProcess Completed ...')
