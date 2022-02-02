# ------------------------------------------------------------------------
# Author:       Jake Pencharz
# Description:  Methods and classes used for an encoder-decoder model
# Date:         July 2019
# Project:      Translate English to German, UCT Final year project
#
# ** Based on tensorflow tutorial:
# https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
#
# Note: if wanting to change the voacab size  and include UNK symbol you  must:
# - https://github.com/keras-team/keras/issues/8092#issuecomment-372833486
# - change embedding layer of the encoder and decoder to vocab size + 1
# - change decoder dense layer size
# ------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.compat.v1.enable_eager_execution()


class Encoder(tf.keras.Model):
    """Encoder Class."""

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, language):
        """Initialise Encoder

        vocab_size - input language vocab size for embedding layer dimensions
        embedding_dim - size of the higher dimensional embedding space
        enc_units - the hidden layer width
        batch_sz - batch size
        langauge - language choice (some models have different configurations)
        """

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units

        if language == "ge2":
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim)
        else:
            # ge3 should only require vocab_size since the unknown token was not added onto the end
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Note: default activation is tanh
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,  # returns output at each step
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    # Note: The GRU processes a batch at a time, therefore it returns a
    # 3 dimensional output tensor (batch_size, sequence length, units)
    # which captures the hidden state of the GRU at each timestep,
    # for every sentence in the batch
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    """Attention Class."""

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """Decoder Class."""

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, language):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units

        if language == "ge2":
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        if language == "ge2":
            self.fc = tf.keras.layers.Dense(vocab_size + 1)
        else:
            self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights
