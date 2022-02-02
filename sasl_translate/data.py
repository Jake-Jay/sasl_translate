import io
import re
import logging
import unicodedata
import tensorflow as tf

# ------------------------------------------------------------------------
# Dataset creation functions
# ------------------------------------------------------------------------


def unicode_to_ascii(s):
    """Converts the unicode file to ascii."""

    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def preprocess_sentence(sentence):
    """Cleans a sentence and adds start and end tokens."""

    sentence = unicode_to_ascii(sentence.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    # sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence) #DGS was trained without ß
    sentence = re.sub(r"[^a-zA-Zß?.!,]+", " ", sentence)

    sentence = sentence.rstrip().strip()

    # adding a start and an end token to the sentence
    sentence = "<start> " + sentence + " <end>"
    return sentence


def create_dataset(path, num_examples, max_sequence_length):
    """
    Remove the accents and cleans the sentences.
    Remove any long sentences.

    Keyword arguments:
    path -- path to file text file
    num_examples -- number of sentence pairs to read from input file
    max_sequence_lenth -- maximum allowed length of a sentence

    Returns:
    sentence pairs -- sentence pairs in the format: [INPUT, TARGET]
    """
    # read lines from file
    lines = io.open(path, encoding="UTF-8").read().strip().split("\n")

    # generate cleaned setence pairs
    sentence_pairs = [
        [preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]
    ]

    logging.info("Sentence pairs before trim: {}".format(len(sentence_pairs)))
      
    # remove any sentences that are too long - for memory's sake 
    short_sentence_pairs = []
    for i, (inp,targ) in enumerate(sentence_pairs):
      if( len(inp.split(' ')) <= max_sequence_length and len(targ.split(' ')) <= max_sequence_length ):
        short_sentence_pairs.append(sentence_pairs[i])
        
    logging.info("Sentence pairs after trim: {}".format(len(short_sentence_pairs)))

    return zip(*short_sentence_pairs)


def tokenize(lang):
    """ 
    Keyword arguments::
    lang -- this is an array of sentences from a single language
    
    Returns:
    tensor -- this is the set of those same sentences but vectorised (each word is now represented by its unique id)
    lang_tokenizer -- language tokeniser object containing closed vocabulary of lang
    """

    if(language_choice == 'ge2' ):
        num_words = 15000
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<unk>", num_words=num_words+1)

        # Creates an internal vocabulary for lang
        lang_tokenizer.fit_on_texts(lang)
        logging.info('Vocab size before: {}'.format( len(lang_tokenizer.word_index) + 1))
        lang_tokenizer.word_index = {e:i for e,i in lang_tokenizer.word_index.items() if i <= num_words}
        lang_tokenizer.word_index[lang_tokenizer.oov_token] = num_words + 1
    elif(language_choice == 'ge3' or language_choice == 'af2'):
        num_words = 15001
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="unk", num_words=num_words)
        lang_tokenizer.fit_on_texts(lang)        
        lang_tokenizer.word_index = {e:i for e,i in lang_tokenizer.word_index.items() if i <= num_words}
        lang_tokenizer.index_word = {i:e for i,e in lang_tokenizer.index_word.items() if i <= num_words}
        

    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="")

        # Creates an internal vocabulary for lang
        lang_tokenizer.fit_on_texts(lang)

    # Uses the tokeniser to vectorise the entire input language
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # Pads all the sequences in the tensor to the same length (max sentence length)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

    return tensor, lang_tokenizer


def load_dataset(path, max_sequence_length, num_examples=None, language=None):
    """Loading cleaned input, output sentence pairs

    Keyword arguments:
    path -- path to file
    max_sequence_lenth -- maximum allowed length of a sentence
    num_examples -- limit on number of sentence pairs to read from input file (default=None)


    Returns: 
    input_tensor -- all input langauge sentences vectorised
    target_tensor -- all target langauge sentences vectorised
    inp_lang_tokeniser -- tokeniser object containing closed vocabulary of input language
    targ_lang_tokeniser -- tokeniser object containing closed vocabulary of target language
    """

    logging.info('Loading the dataset.')

    # Allow the other functions in this file to access the language parameter
    global language_choice
    language_choice = language

    # get sentence pairs
    inp_lang, targ_lang = create_dataset(path, num_examples, max_sequence_length)

    # tokenise/vectorise sentence pairs
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

def max_length(tensor):
    """Finds the longest tensor"""
    return max(len(t) for t in tensor)

def convert(lang, tensor):
    """Method to convert each index in a tensor to its associated word."""
    for unique_id in tensor:
        if unique_id!=0:
            print ("%d \t----> %s" % (unique_id, lang.index_word[unique_id]))
