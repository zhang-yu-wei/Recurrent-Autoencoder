# -*- coding: utf-8 -*-

"""
This file is implemented in order to change the size of the words
"""
import numpy as np
import os.path
from pathlib import *
from itertools import chain, repeat, islice

# get home path
home = str(Path.home())
max_file_index = 133885


# create the index matrix
def create_index_matrix(size):
    indices = []
    for i in range(2**size):
        indices.append([int(x) for x in bin(i)[2:]])
    for i in range(len(indices)):
        indices[i] = [0 for _ in range(size - len(indices[i]))] + indices[i]
    return indices


def save_word(word, path):
    np.savez(path, word)


index_matrix = create_index_matrix(5)
# find the index of the word
for index in range(1, max_file_index + 1):
    path = home + '/yuwei/QM9_smiles/dsgdb9nsd_' + str(index).zfill(6) + '_0' + '.npz'
    # check if that path exists
    if os.path.isfile(path):
        # word will be read in as a numpy array
        word = np.load(path)['s']
        numbers = word.argmax(axis=1)
        word = np.asarray([index_matrix[i] for i in numbers])
        save_path =  home + '/yuwei/smiles/' + str(index).zfill(6) + '.npz'
        save_word(word, save_path)
        print(str(index) + ' word has been saved')






