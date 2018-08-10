# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import logging


class autoencoder(object):
    """
    Class that encapsulates the encoder-decoder architecture
    """

    def __init__(self, hidden_size, num_d_layers, go, train=True, bidirectional=True):
        """
                Initialize the encoder/decoder and creates Tensor objects

                :param hidden_size: hidden layer size of that lstm
                :param go: index of the GO symbol in the embedding matrix
                :param train_embeddings: whether to adjust embeddings during training
                :param bidirectional: whether to create a bidirectional autoencoder
                    (if False, a simple linear LSTM is used)
                """
        # EOS and GO share the same symbol. Only GO needs to be embedded, and
        # only EOS exists as a possible network output
        self.go = go
        self.eos = go

        self.bidirectional = bidirectional

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # the sentence is the object to be memorized
        self.sentence = tf.placeholder(tf.int32, [None, None, 59, 23], 'sentence')
        self.sentence_size = tf.placeholder(tf.int32, [None],
                                            'sentence_size')
        self.l2_constant = tf.placeholder(tf.float32, name='l2_constant')
        self.clip_value = tf.placeholder(tf.float32, name='clip')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.dropout_keep = tf.placeholder(tf.float32, name='dropout_keep')

        self.decoder_step_input = tf.placeholder(tf.int32,
                                                 [None],
                                                 'prediction_step')

        with tf.variable_scope('autoencoder') as self.scope:
            initializer = tf.glorot_normal_initializer()

            # encoder cell
            self.lstm_fw = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer)
            self.lstm_bw = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer)

            # decoder cell
            cells_d = []
            for i in range(num_d_layers):
                with tf.variable_scope('decode_' + str(i)):
                    cells_d.append(tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, activation=tf.nn.softplus))
            with tf.variable_scope('decode_final'):
                cells_d.append(tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, activation=tf.nn.softplus))

            with tf.variable_scope('multi_d'):
                self.stacked_lstm_d = tf.contrib.rnn.MultiRNNCell(cells_d)

            self.initial_d_state = self.stacked_lstm_d.zero_state(hidden_size, tf.float32)

            sent_drpot = tf.nn.dropout(self.sentence, self.dropout_keep)

            # encoding step
            if bidirectional:
                ret = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw, self.lstm_bw, sent_drpot, dtype=tf.float32,
                                                      sequence_length=self.sentence_size, scope=self.scope)
            else:
                ret = tf.nn.dynamic_rnn(self.lstm_fw, sent_drpot, dtype=tf.float32, sequence_length=
                                        self.sentence_size, scope=self.scope)
            _, self.encoded_state = ret

            if bidirectional:
                encoded_state_fw, encoded_state_bw = self.encoded_state

                # set the scope name used inside the decoder
                fw_scope_name = self.scope.name + '/fw'
                bw_scope_name = self.scope.name + '/bw'
            else:
                encoded_state_fw = self.encoded_state
                fw_scope_name = self.scope

            self.scope.reuse_variables()

            # generate a batch of GOs
            # sentence_size has the batch dimension
            go_batch = self._generate_batch_go(tf.size(self.sentence_size))

    def _generate_batch_go(self, batch_size):
        """
        Generate a 1-d tensor with copies of EOS as big as the batch size
        :param like: a tensor whose shape the returned embeddings should match
        :return: a tensor
        """
        ones = tf.ones(batch_size, 59, 23)
        return ones * self.go





