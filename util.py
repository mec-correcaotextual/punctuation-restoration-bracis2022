import os
import numpy as np
from sklearn.model_selection import KFold

BASE_DIR = './texts/tedtalk2012/'
lines_dev = np.array([line.strip() for line in open(os.path.join(BASE_DIR, 'tedtalk2012.dev.txt'), 'r', encoding='utf-8').readlines()])
lines_train = np.array([line.strip() for line in open(os.path.join(BASE_DIR, 'tedtalk2012.train.txt'), 'r', encoding='utf-8').readlines()])
lines_test = np.array([line.strip() for line in open(os.path.join(BASE_DIR, 'tedtalk2012.test.txt'), 'r', encoding='utf-8').readlines()])

kf = KFold(n_splits=10)

for i, (train_index, _) in enumerate(kf.split(lines_train), 1):
    path = f'./texts/tedtalk2012/dataset{i}/'
    os.makedirs(path)
    np.savetxt(os.path.join(path, 'train.txt'), lines_train[train_index], fmt='%s', encoding='utf-8')

for i, (dev_index, _) in enumerate(kf.split(lines_dev), 1):
    path = f'./texts/tedtalk2012/dataset{i}/'

    np.savetxt(os.path.join(path, 'dev.txt'), lines_dev[dev_index], fmt='%s', encoding='utf-8')

for i, (test_index, _) in enumerate(kf.split(lines_test), 1):
    path = f'./texts/tedtalk2012/dataset{i}/'

    np.savetxt(os.path.join(path, 'test.txt'), lines_test[test_index], fmt='%s', encoding='utf-8')
