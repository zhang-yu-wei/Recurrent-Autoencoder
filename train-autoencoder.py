# -*- coding: utf-8 -*-

from __future__ import print_function, division

"""
Script for training the autoencoder.
"""

import tensorflow as tf
import logging
import numpy as np
import argparse

import utils
import autoencoder


def show_parameter_count(variables):
    """
    Count and print how many parameters there are.
    """
    total_parameters = 0
    for variable in variables:
        name = variable.name

        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        print('{}: {} ({} parameters)'.format(name,
                                              shape,
                                              variable_parametes))
        total_parameters += variable_parametes

    print('Total: {} parameters'.format(total_parameters))


def load_or_create_embeddings(path, vocab_size, embedding_size):
    """
    If path is given, load an embeddings file. If not, create a random
    embedding matrix with shape (vocab_size, embedding_size)
    """
    if path is not None:
        return np.load(path)

    embeddings = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_size))
    return embeddings.astype(np.float32)


if __name__ == '__main__':
    class_ = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=class_)
    parser.add_argument('save_dir', help='Directory to file to save trained '
                                         'model')
    parser.add_argument('-n', help='Embedding size (if embeddings not given)',
                        default=50, dest='embedding_size', type=int)
    parser.add_argument('-u', help='Number of LSTM units (when using a '
                                   'bidirectional model, this is doubled in '
                                   'practice)', default=300,
                        dest='lstm_units', type=int)
    parser.add_argument('-r', help='Initial learning rate', default=0.001,
                        dest='learning_rate', type=float)
    parser.add_argument('-b', help='Batch size', default=100,
                        dest='batch_size', type=int)
    parser.add_argument('-e', help='Number of epochs', default=5000,
                        dest='num_epochs', type=int)
    parser.add_argument('-d', help='Dropout keep probability', type=float,
                        dest='dropout_keep', default=1.0)
    parser.add_argument('-i',
                        help='Number of batches between performance report',
                        dest='interval', type=int, default=1000)
    parser.add_argument('-g', help='gpu number', dest='num_gpus', type=int, default=2)
    parser.add_argument('--embeddings',
                        help='Numpy embeddings file. If not supplied, '
                             'random embeddings are generated.')
    parser.add_argument('vocab', help='Vocabulary file')
    parser.add_argument('train', help='Training set')
    parser.add_argument('valid', help='Validation set')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    wd = utils.WordDictionary(args.vocab)
    embeddings = load_or_create_embeddings(args.embeddings, wd.vocabulary_size,
                                           args.embedding_size)

    logging.info('Reading training data')
    train_data = utils.load_binary_data(args.train)
    logging.info('Reading validation data')
    valid_data = utils.load_binary_data(args.valid)
    logging.info('Creating model')

    model = autoencoder.TextAutoencoder(args.lstm_units,
                                        embeddings, wd.eos_index, args.num_gpus
                                        )
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.InteractiveSession(graph=model.g, config=config)
    sess.run(tf.global_variables_initializer())
    model.g.finalize()
    show_parameter_count(model.get_trainable_variables())
    logging.info('Initialized the model and all variables. Starting training.')
    EPOHCS, losses_tra, losses_val = model.train(sess, args.save_dir, train_data, valid_data, args.batch_size,
                args.num_epochs, args.learning_rate,
                args.dropout_keep, 5.0, report_interval=args.interval)

    f = h5py.File("losses.hdf5", "w")
    f['epochs'] = EPOCHS
    f['train'] = losses_tra
    f['valid'] = losses_val
