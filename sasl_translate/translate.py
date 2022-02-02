# ------------------------------------------------------------------------
# Author:       Jake Pencharz
# Description:  Methods and classes used for an encoder-decoder model
# Date:         July 2019
# Project:      Translate English to German, UCT Final year project
#
# ** Based on tensorflow tutorial:
# https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention
# Original file is located at
#     https://colab.research.google.com/drive/1vB4Pp7-SmdOmFy6umqgwGE5QZ-ozslkN
#
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

"""# Neural Machine Translation with Attention"""

from __future__ import absolute_import, division, print_function, unicode_literals


from nltk.translate.bleu_score import corpus_bleu

import time
import os
import logging
import argparse
import pickle
import numpy as np
import tkinter as tk

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import csv

from sklearn.model_selection import train_test_split
from PIL import ImageTk, Image

from sasl_translate.data import load_dataset, max_length, convert, preprocess_sentence
from sasl_translate.model import Encoder, Decoder, BahdanauAttention


font = {"weight": "normal", "size": 22}
matplotlib.rc("font", **font)

logging.basicConfig(filename="nmt.log", filemode="w", level=logging.INFO)


# ------------------------------------------------------------------------
# Setup GUI
# ------------------------------------------------------------------------
def setup_window(window, input_value, target_value):
    if lang_selected == "ge" or lang_selected == "ge2" or lang_selected == "ge3":
        lang_name = "German"
    elif lang_selected == "af" or lang_selected == "af2":
        lang_name = "Afrikaans"
    elif lang_selected == "dgs":
        lang_name = "DGS"
    elif lang_selected == "sasl":
        lang_name = "SASL"

    # Frames
    welcome_frame = tk.Frame(window)
    translation_frame = tk.Frame(window)
    attention_frame = tk.Frame(window, pady=32)
    welcome_frame.pack()
    translation_frame.pack()

    # Welcome
    welcome_label = tk.Label(welcome_frame, text="McDoogle Translate")
    welcome_label.config(font=("Courier", 44), pady=16)
    welcome_label.pack()

    # Translation
    input_label = tk.Label(translation_frame, text="English Sentence")
    input_label.config(font=("Courier", 18), width=18, anchor="w")
    input_entry = tk.Entry(translation_frame, textvariable=input_value, width=32)
    input_entry.bind("<Return>", gui_translate)
    target_label = tk.Label(translation_frame, text=lang_name + " Sentence")
    target_label.config(font=("Courier", 18), width=18, anchor="w")
    # target_entry = tk.Entry(translation_frame, textvariable=target_value, width=32, state='disabled', disabledforeground='grey')
    target_entry = tk.Entry(
        translation_frame,
        textvariable=target_value,
        width=32,
        disabledforeground="grey",
    )

    input_label.grid(row=0, column=0)
    target_label.grid(row=1, column=0)
    input_entry.grid(row=0, column=1)
    target_entry.grid(row=1, column=1)

    # Attention image
    global image_label

    img = Image.open("./output_plots/" + lang_selected + "/attention_gui.png")
    img = img.resize((480, 480), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    image_label = tk.Label(attention_frame, image=img_tk)
    image_label.image = img_tk
    image_label.pack(side="bottom", fill="both", expand="yes")
    attention_frame.pack()

    return input_entry


# ------------------------------------------------------------------------
# Training functions
# ------------------------------------------------------------------------
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        # Encoder output shape: (batch size, sequence length, units) (64, 14, 1024)
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        # Encoder Hidden state shape: (batch size, units) (64, 1024)
        dec_hidden = enc_hidden

        # Create a <start> for each of the sentences in the batch
        # Decoder output shape: (batch_size, vocab size) (64, 16912)
        dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            # Passing enc_output to the decoder for the attention mechanism
            # and processing an entire batch at once
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # Teacher forcing - feeding the target as the next input
            # Note: ignoring the predictions generated due to teacher forcing
            # Note: that targ is a batch of setences. Therefore take the correct word
            # for each sentence in the batch at this timestep
            # and feed it to the next timestep.
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


@tf.function
def validation_step(inp, targ, enc_hidden):
    loss = 0

    # Encoder output shape: (batch size, sequence length, units) (64, 14, 1024)
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    # Encoder Hidden state shape: (batch size, units) (64, 1024)
    dec_hidden = enc_hidden

    # Insert the index for <start> token for every sentences in the batch
    # Decoder output shape: (batch_size, vocab size) (64, 16912)
    dec_input = tf.expand_dims([targ_lang.word_index["<start>"]] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
        # Passing enc_output to the decoder for the attention mechanism
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions)
        # Find the most likely predicted word from entire vocab at step t (for entire batch)
        predicted_id = tf.argmax(predictions, axis=1)
        # Predicted ID is fed back into the model
        dec_input = tf.expand_dims(predicted_id, 1)

    # Average loss over each prediction in output sequence
    batch_loss = loss / int(targ.shape[1])

    return batch_loss


# ------------------------------------------------------------------------
# Evaluation and Testing functions
# ------------------------------------------------------------------------
def evaluate(sentence):
    """Process unicode string sentence as input to network

    Returns an empty string and zero initialised attention plot if there
    is an out of vocabulary word encountered.
    """

    attention_plot = np.zeros((max_length_targ, max_length_inp))

    # Clean the sentence (remove some characters and split the punctuation)
    sentence = preprocess_sentence(sentence)

    # Split the sentence up into words and then map each word to its word-id
    inputs = []
    for word in sentence.split(" "):
        try:
            inputs.append(inp_lang.word_index[word])
        except KeyError:
            print("[evaluation] Unknown word, {}, replacing with <unk>".format(word))
            logging.warning(
                "[evaluation] Unknown word, {}, replacing with <unk>".format(word)
            )
            # inputs.append(inp_lang.word_index['<unk>'])
            inputs.append(inp_lang.word_index["unk"])

    # Pad the list to maximum input length
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_inp, padding="post"
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

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_out
        )

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1,))
        attention_plot[t] = attention_weights.numpy()

        # Grab the most likely prediction
        predicted_id = tf.argmax(predictions[0]).numpy()

        # Convert id to word and add to the result sentence
        result += targ_lang.index_word[predicted_id] + " "

        if targ_lang.index_word[predicted_id] == "<end>":
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def get_validation_pair(x):
    """Using index x, retrieve the input and target validation pair"""
    end_id = inp_lang.word_index["<end>"]
    start_id = inp_lang.word_index["<start>"]

    inp_sentence = []
    for index in input_tensor_val[x]:
        if index > 0 and index != end_id and index != start_id:
            inp_sentence.append(inp_lang.index_word[index])

    end_id = targ_lang.word_index["<end>"]
    start_id = targ_lang.word_index["<start>"]
    targ_sentence = []
    for index in target_tensor_val[x]:
        if index > 0 and index != end_id and index != start_id:
            targ_sentence.append(targ_lang.index_word[index])

    return inp_sentence, targ_sentence


def validate(key):
    inp, targ = get_validation_pair(key)

    print("Ground truth sentence: {}".format(" ".join(targ)))
    result, sentence, attention_plot = evaluate(" ".join(inp))
    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))
    print("Ground truth: {}".format(targ))
    print("Predicted translation: {}".format(result.split(" ")[:-2]))


def gt_vs_prediction(key):
    """Returns a pair of tuples - the ground truth and predicted sentence"""
    inp, targ = get_validation_pair(key)
    result, _, _ = evaluate(" ".join(inp))

    if result == "":
        # If there was an out of vocabulary word and an empty string is returned...
        return targ, ""

    return targ, result.split(" ")[:-2]


def estimate_corpus_bleu():
    """Estimates BLUE score for the translations using up to 1000 samples from validation set"""
    list_references = []
    list_hypotheses = []

    range_end = len(input_tensor_val) if (len(input_tensor_val) < 1000) else 1000

    for i in range(0, range_end):
        ground_truth, prediction = gt_vs_prediction(i)

        if prediction != "":
            # If there was not a key error use this sample for the BLEU score
            list_references.append([ground_truth])
            list_hypotheses.append(prediction)

    return corpus_bleu(list_references, list_hypotheses)


# ------------------------------------------------------------------------
# Training and Testing
# ------------------------------------------------------------------------
def initialise_lite(language_choice, batch=64, lr=0.01):
    """Initialise Lite

    Initialises the input and targer vocabulary as well as the encoder and decoder.
    This is used for inference purposes but could also be used for training if you
    want to use the same dataset and langauge models.
    """
    global units
    BATCH_SIZE = batch
    embedding_dim = 256
    units = 1024
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Read in data and dictionary
    data_dir = "./language_data/" + language_choice + "/"
    prebuilt_model_dir = "./prebuilt_models/" + language_choice + "/"

    print("Unpickling the language models.")
    global inp_lang, targ_lang
    with open(data_dir + "inp_lang.dump", "rb") as file:
        inp_lang = pickle.load(file)

    with open(data_dir + "targ_lang.dump", "rb") as file:
        targ_lang = pickle.load(file)

    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    global max_length_inp, max_length_targ
    if language_choice == "ge":
        max_length_inp, max_length_targ = 15, 24
    elif language_choice == "af":
        max_length_inp, max_length_targ = 25, 25
    elif language_choice == "ge2" or language_choice == "ge3":
        max_length_inp, max_length_targ = 25, 25
    elif language_choice == "dgs":
        max_length_inp, max_length_targ = 25, 25
    else:
        max_length_inp, max_length_targ = 25, 25

    # Init encoder and decoder
    global encoder, decoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, language_choice)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, language_choice)

    # Init checkpoint object
    global checkpoint, checkpoint_dir, checkpoint_prefix, manager

    checkpoint_dir = "./checkpoints/" + language_choice
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, encoder=encoder, decoder=decoder
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)


def initialise(language_choice, batch=64, lr=0.01):
    """Extract Data and execute Training Procedure"""

    data_dir = "./language_data/" + language_choice + "/"
    path_to_file = data_dir + language_choice + "_data.txt"

    if language_choice == "ge":
        num_examples = 150000
    else:
        num_examples = 500000

    max_sentence_length = 25

    # Load dataset into vectorised tensors
    print(
        "* Loading a max of {} datapoints from {}...\n".format(
            num_examples, path_to_file
        )
    )
    logging.info(
        "* Loading {} datapoints from {}...\n".format(num_examples, path_to_file)
    )

    global inp_lang, targ_lang
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
        path_to_file, max_sentence_length, num_examples, language_choice
    )

    print("Pickling the input language data.")
    with open(data_dir + "inp_lang.dump", "wb") as file:
        pickle.dump(inp_lang, file)

    print("Pickling the target language data.")
    with open(data_dir + "targ_lang.dump", "wb") as file:
        pickle.dump(targ_lang, file)

    # Calculate max_length of the target tensors before splitting (so model works for both datasets)
    global max_length_targ, max_length_inp
    max_length_targ, max_length_inp = max_length(target_tensor), max_length(
        input_tensor
    )

    # Creating training and validation sets using an 90-10 split
    global input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val
    (
        input_tensor_train,
        input_tensor_val,
        target_tensor_train,
        target_tensor_val,
    ) = train_test_split(input_tensor, target_tensor, test_size=0.1)

    # Find size of vocabulary
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    # Check what the dataset looks like
    print("DATA METRICS:")
    print("--------------------------------------------------")
    print("Size of inp vocabulary: {}".format(vocab_inp_size))
    print("Size of tar vocabulary: {}\n".format(vocab_tar_size))
    print("Length of padded inp sentences: {}".format(max_length_inp))
    print("Length of padded tar sentences: {}\n".format(max_length_targ))
    print("Size of training dataset:  {} sentences".format(len(input_tensor_train)))
    print("Size of validation dataset: {} sentences".format(len(target_tensor_val)))
    print("--------------------------------------------------")
    logging.info("DATA METRICS:")
    logging.info("--------------------------------------------------")
    logging.info("Size of inp vocabulary: {}".format(vocab_inp_size))
    logging.info("Size of tar vocabulary: {}\n".format(vocab_tar_size))
    logging.info("Length of padded inp sentences: {}".format(max_length_inp))
    logging.info("Length of padded tar sentences: {}\n".format(max_length_targ))
    logging.info(
        "Size of training dataset:  {} sentences".format(len(input_tensor_train))
    )
    logging.info(
        "Size of validation dataset: {} sentences".format(len(target_tensor_val))
    )
    logging.info("--------------------------------------------------")

    # Experiment with converting returned tensor to words from the language
    print("LANGUAGE INDEX EXPERIMENT:")
    print("--------------------------------------------------")
    print("Input Language index to word mapping")
    convert(inp_lang, input_tensor_train[10])
    print()
    print("Target Language index to word mapping")
    convert(targ_lang, target_tensor_train[10])
    print("--------------------------------------------------")

    # ------------------------------------------------------------------------
    # Tuning Parameters
    # ------------------------------------------------------------------------

    global BATCH_SIZE, steps_per_epoch, number_val_batches, embedding_dim, optimizer, loss_object, units

    # Size of the dataset
    BUFFER_SIZE = len(input_tensor_train)
    # Batch size
    BATCH_SIZE = batch
    # Number of batches per epoch
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    number_val_batches = len(input_tensor_val) // BATCH_SIZE
    # Embedding dimension for word embeddings (first layer in encoder and decoder model)
    embedding_dim = 256
    # Dimension of the hidden state (and output space) of encoder and decoder
    units = 1024
    # Optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # Loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    # ------------------------------------------------------------------------
    # Tensorflow Dataset creation
    # ------------------------------------------------------------------------
    global dataset, validation_dataset

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)
    ).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)
    ).shuffle(BUFFER_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

    # ------------------------------------------------------------------------
    # Initialise the ENCODER and test
    # ------------------------------------------------------------------------
    # Extract a sample from the database
    # Note that the size is (batch_size, sequence_length)

    global encoder, decoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, language_choice)

    example_input_batch, example_target_batch = next(iter(dataset))
    print(example_input_batch.shape, example_target_batch.shape)

    # Note that the size of the output has 11 dimensions. This is becuase the output at each timestep is being returned (return_sequence = True)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    print("ENCODER EXPERIMENT")
    print("--------------------------------------------------")
    print(
        "Encoder output shape: (batch size, sequence length, units) {}".format(
            sample_output.shape
        )
    )
    print(
        "Encoder Hidden state shape: (batch size, units) {}".format(sample_hidden.shape)
    )
    print("--------------------------------------------------")

    # ------------------------------------------------------------------------
    # Test Attention layer
    # ------------------------------------------------------------------------
    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("ATTENTION LAYER EXPERIMENT")
    print("--------------------------------------------------")
    # Context vector is the same shape as encoder hidden layer (weighted sum)
    print("Context vector shape: (batch size, units) {}".format(context_vector.shape))
    # Each time step (seq_length) is assigned a weight for each batch
    print(
        "Attention weights shape: (batch_size, sequence_length, 1) {}".format(
            attention_weights.shape
        )
    )
    print("--------------------------------------------------")

    # ------------------------------------------------------------------------
    # Experiment with the decoder
    # ------------------------------------------------------------------------
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, language_choice)

    sample_decoder_output, _, _ = decoder(
        tf.random.uniform((batch, 1)), sample_hidden, sample_output
    )

    # Decoder output is the size of the target language vocabulary for each of the
    # inputs it is given (in this case a batch of 64)

    print("DECODER EXPERIMENT")
    print("--------------------------------------------------")
    print(
        "Decoder output shape: (batch_size, vocab size) {}".format(
            sample_decoder_output.shape
        )
    )
    print("--------------------------------------------------")

    # ------------------------------------------------------------------------
    # Set up Checkpoints
    # ------------------------------------------------------------------------
    global checkpoint, checkpoint_dir, checkpoint_prefix, manager

    checkpoint_dir = "./checkpoints/" + language_choice
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, encoder=encoder, decoder=decoder
    )
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

    return


def train(epochs):
    """Start TRAINING - uses global variables!"""

    print("\nStarting the training...")
    logging.info("\nStarting the training...")

    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    EPOCHS = epochs
    batch_losses = []
    training_losses = []
    validation_losses = []

    """Training Loop"""
    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        total_val_loss = 0

        # Process Training Data
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            # batch: a counting variable for the batch number
            # inp:   the set of [BATCH SIZE] vectorised input sentences
            # targ:  the set of [BATCH SIZE] vectorised target sentences

            batch_loss = train_step(inp, targ, enc_hidden)
            batch_losses.append(batch_loss.numpy())
            total_loss += batch_loss

            if batch % 100 == 0:
                print(
                    "Epoch {} Batch {}/{} Loss {:.4f}".format(
                        epoch + 1, batch, steps_per_epoch, batch_loss.numpy()
                    )
                )
                logging.info(
                    "Epoch {} Batch {}/{} Loss {:.4f}".format(
                        epoch + 1, batch, steps_per_epoch, batch_loss.numpy()
                    )
                )

        print("----------------------------------------")
        print("Training time: {:.4f} sec".format(time.time() - start))
        print("Testing model on validation data...")
        logging.info("----------------------------------------")
        logging.info("Training time: {:.4f} sec".format(time.time() - start))
        logging.info("Testing model on validation data...")

        # Validation Losses each epoch
        for (batch, (inp, targ)) in enumerate(
            validation_dataset.take(number_val_batches)
        ):
            val_batch_loss = validation_step(inp, targ, enc_hidden)
            total_val_loss += val_batch_loss

            if batch % 100 == 0:
                print("Validation batch loss {:.4f}".format(val_batch_loss.numpy()))
                logging.info(
                    "Validation batch loss {:.4f}".format(val_batch_loss.numpy())
                )

        total_val_loss = (total_val_loss / number_val_batches).numpy()
        validation_losses.append(total_val_loss)

        # Test model accuracy on the validation data
        print("Estimating BLEU Score...")
        bleu_score = estimate_corpus_bleu()

        # Saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        epoch_loss = total_loss / steps_per_epoch
        training_losses.append(epoch_loss)

        print(
            "Epoch {}: \n\t- Average Epoch Loss: {:.4f} \n\t- Estimate BLEU Score: {:.4f}".format(
                epoch + 1, epoch_loss, bleu_score
            )
        )
        print("\t- Validation Loss: {:.4f}".format(total_val_loss))
        print("\t- Time taken: {:.4f} sec".format(time.time() - start))
        print("----------------------------------------")

        logging.info(
            "Epoch {}: \n\t- Average Epoch Loss: {:.4f} \n\t- Estimate BLEU Score: {:.4f}".format(
                epoch + 1, epoch_loss, bleu_score
            )
        )
        logging.info("\t- Validation Loss: {:.4f}".format(total_val_loss))
        logging.info("\t- Time taken: {:.4f} sec".format(time.time() - start))
        logging.info("----------------------------------------")

        # Write out batch losses to a CSV file
        with open("50_epochs_150000_sentences.losses", "w") as file:
            csv_writer = csv.writer(
                file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(batch_losses[0::100])
            csv_writer.writerow(validation_losses)
            csv_writer.writerow(training_losses)

    print("\nTraining is complete...\n\n")
    logging.info("\nTraining is complete...\n\n")

    # ------------------------------------------------------------------------
    # Save results from testing
    # ------------------------------------------------------------------------
    # Plot and save the loss information
    plt.figure()
    plt.plot(batch_losses)
    plt.xlabel("Batches")
    plt.ylabel("Training Losses")
    plt.savefig("./output_plots/" + lang_selected + "training_losses.png")

    plt.figure()
    val_curve = plt.plot(validation_losses, "r", label="Validation Losses")
    train_curve = plt.plot(
        batch_losses[0::steps_per_epoch], "b", label="Training Losses"
    )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("./output_plots/" + lang_selected + "validation_losses.png")

    # ------------------------------------------------------------------------
    # Run some quick tests after training
    # ------------------------------------------------------------------------
    print("\nStarting the testing...")
    logging.info("\nStarting the testing...")

    # Restoring the latest checkpoint
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Sentences from validation set
    for phrase in range(0, 10):
        validate(phrase)
        print()


def plot_attention(attention, sentence, predicted_sentence, lang_selected):
    """Plot attention weights"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap="viridis")

    fontdict = {"fontsize": 14}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig(
        "./output_plots/"
        + lang_selected
        + "/attention_"
        + "".join(sentence[1:-2])
        + ".png"
    )
    # plt.show()


def gui_plot_attention(attention, sentence, predicted_sentence, lang_selected):
    """Plot attention weights"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(attention[:-1, :], cmap="viridis")
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    fontdict = {"fontsize": 24}

    ax.set_xticklabels([""] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([""] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()
    plt.savefig("./output_plots/" + lang_selected + "/attention_gui.png")
    # plt.show()


def translate_plot_attention(sentence, lang_selected):
    """Uses trained network to translate input sentence and plots attention weights"""
    result, sentence, attention_plot = evaluate(sentence)

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    attention_plot = attention_plot[
        : len(result.split(" ")), : len(sentence.split(" "))
    ]
    plot_attention(
        attention_plot, sentence.split(" "), result.split(" "), lang_selected
    )


def translate(sentence):
    result, sentence_, attention_plot = evaluate(sentence)

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))


def gui_translate(event):
    print("\nEvent Triggered: Translating...")
    sentence = input_entry.get()

    result, sentence, attention_plot = evaluate(sentence)
    # reverse split only twice to remove space at the end and <end> token
    if lang_selected == "sasl" or lang_selected == "dgs":
        target_value.set(result.rsplit(" ", 2)[0].upper())
    else:
        target_value.set(result.rsplit(" ", 2)[0])

    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))

    attention_plot = attention_plot[
        : len(result.split(" ")), : len(sentence.split(" "))
    ]
    if lang_selected == "sasl" or lang_selected == "dgs":
        gui_plot_attention(
            attention_plot,
            sentence.split(" "),
            result.upper().split(" "),
            lang_selected,
        )
    else:
        gui_plot_attention(
            attention_plot, sentence.split(" "), result.split(" "), lang_selected
        )

    img = Image.open("./output_plots/" + lang_selected + "/attention_gui.png")
    img = img.resize((480, 480), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    return result


# ------------------------------------------------------------------------
# Main Method
# ------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        help="required to select language from: \
                        german (ge), afrikaans (af), DGS (dgs), SASL (sasl)",
    )
    parser.add_argument(
        "--train",
        help="toggle training. Requires one to set epochs.",
        action="store_true",
    )
    parser.add_argument("--lr", help="specify learning rate.", type=float)
    parser.add_argument("--epochs", help="specify number of training epochs.", type=int)
    parser.add_argument("--batch", help="specify batch size if training.", type=int)
    parser.add_argument(
        "--gui",
        help="specify if you want to launch GUI. Requires --lang to be set.",
        action="store_true",
    )

    # Collect arguments
    global lang_selected
    args = parser.parse_args()
    lang_selected = args.lang
    translate = args.gui
    train_flag = args.train
    batch = args.batch
    lr = args.lr

    if type(batch) != int:
        print("Setting batch size to 64 by default.")
        logging.info("Setting batch size to 64 by default.")
        batch = 64
    else:
        logging.info("Setting batch size to: ", batch)
        print("Setting batch size to: ", batch)

    if type(lr) != float:
        print("Setting learning rate to default of 0.01.")
        logging.info("Setting learning rate to default of 0.01.")
        lr = 0.01
    else:
        logging.info("Setting lr to: ", lr)
        print("Setting batch size to: ", lr)

    # Train Procedure
    if train_flag:
        epochs = args.epochs

        print(type(epochs))
        if type(epochs) != int:
            print(
                "Required to specify number of epochs if training. Use -h flag for help."
            )
        else:
            print("Selected to train!")
            # Initialise the model
            logging.debug("[main] Initialising the model and vocabulary.")
            initialise(lang_selected, batch, lr)

            print("Restoring checkpoint from {}".format(checkpoint_dir))
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
            else:
                print("Initializing from scratch.")

            train(epochs)

    # Translation and GUI Procedure
    elif translate:
        # Setup the GUI
        window = tk.Tk()
        window.title("McDoogle Translate")
        window.geometry("{}x{}".format(800, 800))

        input_value = tk.StringVar()
        input_value.set("Insert text here")
        target_value = tk.StringVar()

        input_entry = setup_window(window, input_value, target_value)

        # Load the model
        initialise_lite(lang_selected, batch, lr)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        window.mainloop()

    else:
        print("Neither training or testing. Exiting...")
        exit()
