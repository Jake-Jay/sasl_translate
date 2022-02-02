# ------------------------------------------------------------------------
# Author:       Jake Pencharz
# Description:  Methods and classes used for evaluation, visualisation
#               and validation of the model
# Date:         July 2019
# Project:      Translate EN to SASL, UCT Final year project
#
# ** Based on tensorflow tutorial:
# https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------------------------

import numpy as np
import time
import tensorflow as tf

from nltk.translate.bleu_score import corpus_bleu

from sasl_translate.data import preprocess_sentence


def evaluate(
    sentence,
    input_tensor_val,
    target_tensor_val,
    inp_lang,
    targ_lang,
    units,
    encoder,
    decoder,
):
    """
    Description:
    - Uses the encoder to encode the input sentnece. Iteratively uses
    the decoder to create an output
    - Attention weights used at each stage in the decoder are saved to show
    where the decoder focused

    Parameters:
    - sentence: this is a sentence in raw string format

    Returns:
    - attention_plot: matrix of attention weights (target_seq_len, inp_seq_len)
    - result: resulting string predicted by decoder
    - sentence: the orginal sentence but preprocessed.
    """

    max_len_input = input_tensor_val.shape[1]
    max_len_target = target_tensor_val.shape[1]
    attention_plot = np.zeros((max_len_target, max_len_input))

    # Clean the sentence (remove some characters and split the punctuation)
    sentence = preprocess_sentence(sentence)

    # Split the sentence up into words and then map each word to its word-id
    inputs = [inp_lang.word_index[word] for word in sentence.split(" ")]

    # Pad the list to maximum input length
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_len_input, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    hidden = [tf.zeros((1, units))]

    # Pass input to encoder
    enc_out, enc_hidden = encoder(inputs, hidden)

    # Initialise decoder hidden state with the encoder's and start
    dec_hidden = enc_hidden

    # Add a batch dimension to a single element
    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]], 0)

    for t in range(max_len_target):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        # Grab the most likely prediction
        predicted_id = tf.argmax(predictions[0]).numpy()

        # If padding is predicted, end the sentence
        if predicted_id == 0:
            predicted_id = 2

        # Convert id to word and add to the result sentence
        result += targ_lang.index_word[predicted_id] + " "

        if targ_lang.index_word[predicted_id] == "<end>":
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def validate(key, input_tensor_val, target_tensor_val, inp_lang, targ_lang):
    inp, targ = get_validation_pair(
        key, input_tensor_val, target_tensor_val, inp_lang, targ_lang
    )

    print("Ground truth sentence: {}".format(" ".join(targ)))
    result, sentence, attention_plot = evaluate(" ".join(inp))
    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))
    print("Ground truth: {}".format(targ))
    print("Predicted translation: {}".format(result.split(" ")[:-2]))


def get_validation_pair(x, input_tensor_val, target_tensor_val, inp_lang, targ_lang):
    """Using index x, retrieve the input and target validation pair"""
    inp_sentence = []
    for index in input_tensor_val[x]:
        if index > 2:
            inp_sentence.append(inp_lang.index_word[index])

    targ_sentence = []
    for index in target_tensor_val[x]:
        if index > 2:
            targ_sentence.append(targ_lang.index_word[index])

    return inp_sentence, targ_sentence


def gt_vs_prediction(
    key,
    input_tensor_val,
    target_tensor_val,
    inp_lang,
    targ_lang,
    units,
    encoder,
    decoder,
):
    """
    Returns a pair of lists containing the tokenised ground truth
    and predicted sentences.
    """
    inp, targ = get_validation_pair(
        key, input_tensor_val, target_tensor_val, inp_lang, targ_lang
    )
    start = time.time()
    result, sentence, attention_plot = evaluate(
        " ".join(inp),
        input_tensor_val,
        target_tensor_val,
        inp_lang,
        targ_lang,
        units,
        encoder,
        decoder,
    )
    print("Sentence: {}, time: {}".format(result, time.time() - start))
    return targ, result.split(" ")[:-2]


def estimate_corpus_bleu(
    input_tensor_val, target_tensor_val, inp_lang, targ_lang, units, encoder, decoder
):
    """
    Estimates BLEU score.
    Uses up to 1000 samples from validation set.
    """
    list_references = []
    list_hypotheses = []

    range_end = len(input_tensor_val) if (len(input_tensor_val) < 1000) else 1000

    for i in range(0, range_end):
        ground_truth, prediction = gt_vs_prediction(
            i,
            input_tensor_val,
            target_tensor_val,
            inp_lang,
            targ_lang,
            units,
            encoder,
            decoder,
        )
        list_references.append([ground_truth])
        list_hypotheses.append(prediction)

    return corpus_bleu(list_references, list_hypotheses)
