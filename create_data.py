# -*- coding: utf-8 -*-

"""This file is used for creating sentences"""
import numpy as np
from pathlib import *
import random
import os.path
import argparse


def generate_sents(num_sents=10000, num_words=100, num_max_length=20, valid_prop=0.01):
    """
                    generate sentences using the words given

                    :param num_sents: number of example sentences we want
                    :param num_words: number of words exist in real
                    :param num_max_length: maximum length of the sentence
                    :param valid_prop: data proportion for validation
                    """
    train_data = {}  # set up a dict to memorize training data
    valid_data = {}  # set up a dict to memorize validation data
    sents = []  # memorize sentences
    sizes = []  # memorize lengths of the sentences
    home = str(Path.home())  # home path
    num_max_words = 133885  # the maximum index number
    # generate a word list
    vocabulary = {}
    while len(vocabulary) < num_words:
        index = random.randint(1, num_max_words)
        path = home + '/yuwei/smiles/' + str(index).zfill(6) + '.npz'
        # check if that path exists
        if os.path.isfile(path):
            word = np.load(path)['s']
            vocabulary['%d' % index] = word

    for i in range(num_sents):
        # randomly choose a sentence length
        length = random.randint(2, num_max_length)
        # set up a list to memorize sentence
        sentence = random.sample(list(vocabulary), length)
        print('sentence ' + str(i + 1) + ' has ' + str(length) + ' words.')
        sents.append(sentence)
        sizes.append(length)

    ind = int(num_sents * valid_prop)
    train_data['sentences'] = sents[ind:]
    train_data['sizes'] = sizes[ind:]
    valid_data['sentences'] = sents[:ind]
    valid_data['sizes'] = sizes[:ind]

    return list(vocabulary), train_data, valid_data


def write_vocabulary(words, path):
    """
    Write the contents of word_dict to the given path.
    """
    text = '\n'.join(words)
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', help='the path you want to save the data')
    parser.add_argument('--num-words', help='number of words you want to use', default=100, dest='num_words')
    parser.add_argument('--num-sents', help='number of sentences you want to generate', default=10000, dest='num_sents')
    parser.add_argument('--max-len', help='maximum length of sentences', default=20, dest='num_max_length')
    parser.add_argument('--valid-prop', help='proportion of valid data in whole dataset', default=0.01,
                        dest='valid_prop')
    args = parser.parse_args()

    vocabulary, train_data, valid_data = generate_sents(args.num_sents, args.num_words,
                                                        args.num_max_length, args.valid_prop)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    path = os.path.join(args.output, 'vocabulary.txt')
    write_vocabulary(vocabulary, path)

    path = os.path.join(args.output, 'train-data.npz')
    np.savez(path, **train_data)

    path = os.path.join(args.output, 'valid-data.npz')
    np.savez(path, **valid_data)



