from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

corpus: ColumnCorpus = ColumnCorpus('./data/tedtalk2012/', column_format={0: 'text', 1: 'ner'})

tagger: SequenceTagger = SequenceTagger.load('ner')
result = tagger.evaluate(corpus.test)
