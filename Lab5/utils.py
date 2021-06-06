import keras
import numpy as np
from nltk.tokenize import TreebankWordTokenizer as twt
from os import listdir
from xml.dom.minidom import parse
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import tensorflow as tf
import pickle
from keras.models import Model, load_model
from keras_contrib.layers import CRF
from keras.initializers import RandomUniform
import tensorflow.keras.backend as K
from keras.layers import TimeDistributed, Reshape, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from keras_contrib.utils import save_load_utils


############### load_data function ###############
def tokenize(s):
    """
    Task:
    Given a sentence, calls nltk.tokenize to split it in
    tokens, and adds to each token its start/end offset
    in the original sentence.
    Input:
    s: string containing the text for one sentence
    Output:
    Returns a list of tuples (word, offsetFrom, offsetTo)
    Example:
    tokenize("Ascorbic acid, aspirin, and the common cold.")
    [("Ascorbic", 0, 7), ("acid", 9, 12) , (",", 13, 13),
    ("aspirin", 15, 21), (",", 22, 22), ("and", 24, 26),
    ("the", 28, 30), ("common", 32, 37), ("cold", 39, 42),
    (".", 43, 43)]
    """

    # span_tokenize identifies the tokens using integer offsets: (start_i, end_i)
    list_offset = list(twt().span_tokenize(s))

    # Create the list of tuples of each token and its start/end offset
    tokens = [(s[list_offset[i][0]:list_offset[i][1]], list_offset[i][0], list_offset[i][1]-1) for i in range(len(list_offset))]
    return tokens


def get_tag(token, gold):
    """ Task:
            Given a token and a list of ground truth entities in a sentence, decide which is the B-I-O tag for the token
        Input:
            token: A token, i.e. one triple (word, offsetFrom, offsetTo)
            gold: A list of ground truth entities, i.e. a list of triples (offsetFrom, offsetTo, type)
        Output:
            The B-I-O ground truth tag for the given token ("B-drug", "I-drug", "B-group", "I-group", "O", ...)
        Example:
            >> get_tag((" Ascorbic ", 0, 7), [(0, 12, "drug"), (15, 21, "brand")])
            B-drug
            >> get_tag ((" acid ", 9, 12), [(0, 12, "drug"), (15, 21, "brand ")])
            I-drug
            >> get_tag ((" common ", 32, 37), [(0, 12, "drug"), (15, 21, "brand")])
            O
            >> get_tag ((" aspirin ", 15, 21), [(0, 12, "drug"), (15, 21, "brand ")])
            B-brand
    """

    offset_B = [gold[i][0] for i in range(len(gold))]
    offset_L = [gold[i][1] for i in range(len(gold))]
    offset_int = [(offset_B[i], offset_L[i]) for i in range(len(gold))]

    if token[1] in offset_B:
        index = [x for x, y in enumerate(gold) if y[0] == token[1]]
        tag = "B-" + str(gold[index[0]][2])
    elif token[2] in offset_L:
        index = [x for x, y in enumerate(gold) if y[1] == token[2]]
        tag = "I-" + str(gold[index[0]][2])
    else:
        flag = 0
        for inter in offset_int:
            if token[1] > inter[0] and token[2] <= inter[1]:
                index = [x for x, y in enumerate(gold) if y[0] == inter[0]]
                tag = "I-" + str(gold[index[0]][2])
                flag = 1
        if flag == 0:
            tag = "O"
    return tag


def load_data(data_dir):
    """
    Task :
    Load XML files in given directory , tokenize each sentence , and extract
    ground truth BIO labels for each token .

    Input :
    datadir : A directory containing XML files .

    Output :
    A dictionary containing the dataset . Dictionary key is sentence_id , and
    the value is a list of token tuples (word , start , end , ground truth ).

    Example :
    >> load_data('data/Train')
    {'DDI - DrugBank . d370 .s0 ': [(' as ', 0, 1,'O '), (' differin ',3,10,'B- brand '),
    (' gel ',12,14,'O '), ... , (' with ' ,343 ,346 , 'O '),
    (' caution ' ,348 ,354 , 'O '), ( '. ' ,355 ,355 , 'O ')],
    'DDI - DrugBank . d370 .s1 ': [(' particular ',0,9,'O '), (' caution ',11,17,'O '),
    (' should ',19,24,'O '), ... ,( ' differin ' ,130 ,137 , 'B- brand '),
    (' gel ',139, 141 ,'O '), ( '. ' ,142 ,142 , 'O ')],
    ...
    }
    """

    # Initialize dictionary to return parsed data
    parsed_data = {}
    # Initialize max_len to 0
    max_length = 0

    # process each file in directory
    for f in listdir(data_dir):
        # parse XML file, obtaining a DOM tree
        tree = parse(data_dir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            s_text = s.attributes["text"].value  # get sentence text
            # load  ground  truth  entities.
            gold = []
            entities = s.getElementsByTagName("entity")
            for e in entities:
                # for  discontinuous  entities , we only  get  the  first  span
                offset = e.attributes["charOffset"].value
                (start, end) = offset.split(";")[0].split("-")
                gold.append((int(start), int(end), e.attributes["type"].value))

            # tokenize text
            tokens = tokenize(s_text)
            # update max_length
            if len(tokens) > max_length:
                max_length = len(tokens)
            # if the sentence is not empty add tokens in the sentence to the dictionary
            if len(tokens) > 0:
                for i in range(0, len(tokens)):
                    # see if the token is part of an entity , and which part (B/I)
                    tag = get_tag(tokens[i], gold)
                    tokens[i] = tokens[i] + (tag,)
                parsed_data[sid] = tokens
    return parsed_data, max_length


############### create_indexes function ###############
def create_indexes(dataset, max_length):
    """
    Task :
    Create index dictionaries both for input (words) and output (labels)
    from given dataset.
    Input :
    dataset : dataset produced by load_data.
    max_length : maximum length of a sentence (longer sentences will
    be cut, shorter ones will be padded).
    Output :
    A dictionary where each key is an index name (e.g. "words", "labels"),
    and the value is a dictionary mapping each word / label to a number.
    An entry with the value for max_len is also stored
    Example :
    >> create_indexes(train_data)
    {' words ': {'<PAD > ':0, '<UNK >':1 , '11- day ':2, 'murine ':3, 'criteria ':4,
    'stroke ':5 ,... ,' levodopa ':8511 , 'terfenadine ': 8512}
    'labels ': {'<PAD > ':0 , 'B- group ':1, 'B- drug_n ':2, 'I- drug_n ':3, 'O ':4,
    'I- group ':5, 'B- drug ':6, 'I- drug ':7, 'B- brand ':8, 'I- brand ':9}
    'max_len ' : 100
    }
    """

    lemmatizer = WordNetLemmatizer()

    # Initialize dictionaries for each type of information with the corresponding indexes for Padding and Unknown values
    words = {"<PAD>": 0, "<UNK>": 1}
    idx_words = 2
    lemmas = {"<PAD>": 0, "<UNK>": 1}
    idx_lemmas = 2
    pos = {"<PAD>": 0, "<UNK>": 1}
    idx_pos = 2
    suffixes = {"<PAD>": 0, "<UNK>": 1}
    idx_suffixes = 2
    prefixes = {"<PAD>": 0, "<UNK>": 1}
    idx_prefixes = 2
    labels = {"<PAD>": 0}
    idx_labels = 1

    for sid, sentence in dataset.items():
        if len(sentence) < max_length:
            # Extract lemmas and PoS tags of the current sentence
            sentence_words = [sentence[i][0] for i in range(len(sentence))]
            sentence_lemmas = [lemmatizer.lemmatize(token[0].lower()) for token in sentence]
            sentence_pos = [pos_tag(sentence_words)[i][1] for i in range(len(sentence_words))]
            # Add elements to the dictionaries if they still do not exist
            for index, token in enumerate(sentence):
                if token[0].lower() not in words:
                    words[token[0].lower()] = idx_words
                    idx_words += 1
                if sentence_lemmas[index] not in lemmas:
                    lemmas[sentence_lemmas[index]] = idx_lemmas
                    idx_lemmas += 1
                if sentence_pos[index] not in pos:
                    pos[sentence_pos[index]] = idx_pos
                    idx_pos += 1
                if token[0][-5:].lower() not in suffixes and len(token[0][-5:]) == 5:
                    suffixes[token[0][-5:].lower()] = idx_suffixes
                    idx_suffixes += 1
                if token[0][:4].lower() not in prefixes and len(token[0][:4]) == 4:
                    prefixes[token[0][:4].lower()] = idx_prefixes
                    idx_prefixes += 1
                if token[3] not in labels:
                    labels[token[3]] = idx_labels
                    idx_labels += 1
    # Return the definitive dictionary with all the information retrieved
    return {"words": words, "lemmas": lemmas, "pos": pos, "suffixes": suffixes, "prefixes": prefixes,
            "labels": labels, "max_len": max_length}


############### build_network function ###############
def build_network(idx):
    """
    Task : Create network for the learner.
    Input :
    idx : index dictionary with word/labels codes, plus maximum sentence length.
    Output :
    Returns a compiled Keras neural network with the specified layers
    """
    # sizes

    n_words = len(idx['words'])
    n_lemmas = len(idx['lemmas'])
    n_pos = len(idx['pos'])
    n_suffixes = len(idx['suffixes'])
    n_labels = len(idx['labels'])
    max_len = idx['max_len']

    '''# create network layers ESTE ESTA BIEEEEEN
    inp = Input(shape=(max_len,4))
    #model = Reshape((2 * max_len, 1), input_shape=(
    #    max_len, 4))
    model = Embedding(input_dim=n_words + 1, output_dim=100,input_length=(max_len,4), mask_zero=False)(inp)  # 20-dim embedding
    #model = Reshape((max_len, 200))(model)
    model=Reshape((max_len, 400, 1))(model)
    newdim = tuple([x for x in model.shape.as_list() if x != 1 and x is not None])
    reshape_layer = Reshape(newdim)(model)
    #model = model[:,:,:,0]
    model = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.1))(reshape_layer)  # variational biLSTM
    model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_labels,sparse_target=True)  # CRF layer
    out = crf(model)
    model = Model(inp, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())''' # HASTA AQUI ESTA BIEN Y FUNCIONA PRIMER MODELO
    '''#TODO:REVISAR ESTO CREO QUE CHARACTER INPUT SOBRA
    char2Idx = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>":
        char2Idx[c] = len(char2Idx)
    character_input = Input(shape=(None, 52,), name="Character_input")
    embed_char_out = TimeDistributed(
        Embedding(len(char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name="Character_embedding")(
        character_input)

    dropout = Dropout(0.5)(embed_char_out)

    # CNN
    conv1d_out = TimeDistributed(
        Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1),
        name="Convolution")(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52), name="Maxpool")(conv1d_out)
    char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
    char = Dropout(0.5)(char)

    # word-level input
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],
                      weights=[wordEmbeddings],
                      trainable=False)(words_input)

    # case-info input
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                       weights=[self.caseEmbeddings],
                       trainable=False)(casing_input)

    # concat & BLSTM
    output = concatenate([words, casing, char])
    output = Bidirectional(LSTM(50,
                                return_sequences=True,
                                dropout=0.5,  # on input to each LSTM block
                                recurrent_dropout=0.1  # on recurrent input signal
                                ), name="BLSTM")(output)
    output = TimeDistributed(Dense(len(label2Idx), activation='softmax'), name="Softmax_layer")(output)

    # set up model
    model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])

    model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)'''
    inp0 = Input(shape=(max_len,))
    inp1 = Input(shape=(max_len,))
    #inp2 = Input(shape=(max_len,))
    #inp3 = Input(shape=(max_len,))
    emb1 = Embedding(input_dim=n_words + 1, output_dim=2000, input_length=(max_len,), mask_zero=False)(
        inp0)  # 20-dim embedding
    emb2 = Embedding(input_dim=n_words + 1, output_dim=50, input_length=(max_len,), mask_zero=False)(
        inp1)  # 20-dim embedding
    #emb3 = Embedding(input_dim=n_words + 1, output_dim=500, input_length=(max_len,), mask_zero=False)(
    #    inp2)  # 20-dim embedding
    #emb4 = Embedding(input_dim=n_words + 1, output_dim=50, input_length=(max_len,), mask_zero=False)(
    #    inp3)  # 20-dim embedding
    combined = concatenate([emb1, emb2])
    model = Bidirectional(LSTM(units=250, input_shape=emb1.shape, return_sequences=True,
                               recurrent_dropout=0.1), input_shape=emb1.shape)(combined)  # variational biLSTM
    model = TimeDistributed(Dense(250, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_labels, sparse_target=False)  # CRF layer
    out = crf(model)
    model = Model(inputs=[inp0,inp1], outputs=out)
    model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())
    return model
    '''inp = Input(shape=(max_len, 4))
    # model = Reshape((2 * max_len, 1), input_shape=(
    #    max_len, 4))
    model = Embedding(input_dim=n_words + 1, output_dim=100, input_length=(max_len, 4), mask_zero=False)(
        inp)  # 20-dim embedding
    # model = Reshape((max_len, 200))(model)
    model = Reshape((max_len, 400, 1))(model)
    newdim = tuple([x for x in model.shape.as_list() if x != 1 and x is not None])
    reshape_layer = Reshape(newdim)(model)
    # model = model[:,:,:,0]
    model = Bidirectional(LSTM(units=250, return_sequences=True,
                               recurrent_dropout=0.1))(reshape_layer)  # variational biLSTM
    model = TimeDistributed(Dense(250, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(n_labels, sparse_target=True)  # CRF layer
    out = crf(model)
    model = Model(inp, out)
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    print(model.summary())'''


############### encode_words function ###############
def encode_words(dataset, idx):
    """
    Task :
    Encode the words in a sentence dataset formed by lists of tokens into
    lists of indexes suitable for NN input.

    Input :
    dataset : A dataset produced by load_data.
    idx : A dictionary produced by create_indexes, containing word and
    label indexes, as well as the maximum sentence length.

    Output :
    The dataset encoded as a list of sentences, each of them is a list of
    word indices. If the word is not in the index, <UNK> code is used. If
    the sentence is shorter than max_len it is padded with <PAD > code.

    Example :
    >> encode_words(train_data, idx)
    [ [6882 1049 4911 ... 0 0 0 ]
    [2290 7548 8069 ... 0 0 0 ]
    ...
    [2002 6582 7518 ... 0 0 0 ] ]
    """

    lemmatizer = WordNetLemmatizer()

    # Initialize an empty list for the encoded information of the sentences
    encoded_words = []
    # For each sentence encode the information in its words (words, lemmas, PoS tags, ...)
    for sid, sentence in dataset.items():
        if len(sentence) < idx["max_len"]:
            # Extract lemmas and PoS tags of the current sentence
            sentence_words = [sentence[i][0] for i in range(len(sentence))]
            sentence_lemmas = [lemmatizer.lemmatize(token[0].lower()) for token in sentence]
            sentence_pos = [pos_tag(sentence_words)[i][1] for i in range(len(sentence_words))]

            # Encode
            this_words = [idx["words"][word[0].lower()] if word[0].lower() in idx["words"]
                          else idx["words"]["<UNK>"] for word in sentence]
            this_lemmas = [idx["lemmas"][lemma] if lemma in idx["lemmas"]
                           else idx["lemmas"]["<UNK>"] for lemma in sentence_lemmas]
            this_pos = [idx["pos"][pos] if pos in idx["pos"]
                        else idx["pos"]["<UNK>"] for pos in sentence_pos]
            this_suffixes = [idx["suffixes"][word[0][-5:].lower()] if word[0][-5:].lower() in idx["suffixes"]
                             else idx["suffixes"]["<UNK>"] for word in sentence]
            this_prefixes = [idx["prefixes"][word[0][:4].lower()] if word[0][:4].lower() in idx["prefixes"]
                             else idx["prefixes"]["<UNK>"] for word in sentence]

            this_padding = [idx["words"]["<PAD>"] for _ in range(idx["max_len"] - len(sentence))]
            this_words.extend(this_padding)
            this_lemmas.extend(this_padding)
            this_pos.extend(this_padding)
            this_suffixes.extend(this_padding)
            this_prefixes.extend(this_padding)

            this_sentence_info = [list(elem) for elem
                                  in zip(this_words, this_lemmas, this_pos, this_suffixes, this_prefixes)]
            encoded_words.append(this_sentence_info)

    return encoded_words


############### encode_labels function ###############
def encode_labels(dataset, idx):
    """
    Task :
    Encode the ground truth labels in a sentence dataset formed by lists of
    tokens into lists of indexes suitable for NN output .

    Input :
    dataset : A dataset produced by load_data .
    idx : A dictionary produced by create_indexes, containing word and
    label indexes, as well as the maximum sentence length .

    Output :
    The dataset encoded as a list of sentence, each of them is a list of
    BIO label indices. If the sentence is shorter than max_len it is
    padded with <PAD> code .

    Example :
    >> encode_labels(train_data, idx)
    [[ [4] [6] [4] [4] [4] [4] ... [0] [0] ]
    [ [4] [4] [8] [4] [6] [4] ... [0] [0] ]
    ...
    [ [4] [8] [9] [4] [4] [4] ... [0] [0] ]
    ]
    """

    encoded_labels = []
    for sid, sentence in dataset.items():
        if len(sentence) < idx["max_len"]:
            this_labels = [[idx["labels"][word[3]]] for word in sentence]
            this_padding = [[idx["labels"]["<PAD>"]] for _ in range(idx["max_len"] - len(sentence))]
            this_labels.extend(this_padding)
            encoded_labels.append(this_labels)

    return encoded_labels


############### save_model_and_indexes function ###############
def save_model_and_indexes(model, idx, filename):
    """
    Task : Save given model and indexes to disk
    Input :
    model : Keras model created by _build_network, and trained.
    idx : A dictionary produced by create_indexes, containing word and
    label indexes, as well as the maximum sentence length.
    filename : filename to be created
    Output :
    Saves the model into filename .nn and the indexes into filename .idx
    """
    # save the model    # TODO: esto decía que lo hiciéramos con Keras, no idea "model.save"
    #pickle.dump(model, open("models_and_idxs/" + filename + ".nn", 'wb'))
    #save_load_utils.save_all_weights(model, "models_and_idxs/" + filename + ".nn")
    model.save_weights("models_and_idxs/" + filename + ".nn")

    # save the dictionary of indexes
    pickle.dump(idx, open("models_and_idxs/" + filename + ".idx", 'wb'))


############### load_model_and_indexes function ###############
<<<<<<< HEAD



def load_model_and_indexes(filename):
    """
    Task : Load model and associate indexes from disk
    Input :
    filename : filename to be loaded
    Output :
    Loads a model from filename .nn, and its indexes from filename .idx
    Returns the loaded model and indexes.
    """

    # load the model    # TODO: esto decía que lo hiciéramos con Keras, no idea "keras.models.load model"
    # model = pickle.load(open("models_and_idxs/" + filename + ".nn", 'rb'))
    # model = keras.models.load_model("models_and_idxs/" + filename + ".nn")
    '''inp1 = np.array([[item[0] for item in sublist] for sublist in X_train])
    inp2 = np.array([[item[1] for item in sublist] for sublist in X_train])
    inp3 = np.array([[item[2] for item in sublist] for sublist in X_train])
    inp4 = np.array([[item[3] for item in sublist] for sublist in X_train])
    val1 = np.array([[item[0] for item in sublist] for sublist in X_val])
    val2 = np.array([[item[1] for item in sublist] for sublist in X_val])
    val3 = np.array([[item[2] for item in sublist] for sublist in X_val])
    val4 = np.array([[item[3] for item in sublist] for sublist in X_val])
    Y_train = np.array(
        [[[0.0 if value != item[0] else 1.0 for value in range(len(np.zeros((10,))))] for item in sublist] for sublist
         in Y_train])
    Y_val = np.array(
        [[[0.0 if value != item[0] else 1.0 for value in range(len(np.zeros((10,))))] for item in sublist] for sublist
         in Y_val])'''
<<<<<<< HEAD
    idx = pickle.load(open("models_and_idxs/" + filename + ".idx", 'rb'))
    #model = build_network(idx)
    #model.fit([inp1[0:2],inp2[0:2],inp3[0:2],inp4[0:2]], Y_train[0:2], validation_data=([val1[0:2],val2[0:2],val3[0:2],val4[0:2]], Y_val[0:2]), batch_size = 32, epochs= 4)
    n_words = len(idx['words'])
    n_lemmas = len(idx['lemmas'])
    n_pos = len(idx['pos'])
    n_suffixes = len(idx['suffixes'])
    n_labels = len(idx['labels'])
    max_len = idx['max_len']

    inp0 = Input(shape=(max_len,))
    inp1 = Input(shape=(max_len,))
    # inp2 = Input(shape=(max_len,))
    # inp3 = Input(shape=(max_len,))
    emb1 = Embedding(input_dim=n_words + 1, output_dim=2000, input_length=(max_len,), mask_zero=False)(
        inp0)  # 20-dim embedding
    emb2 = Embedding(input_dim=n_words + 1, output_dim=50, input_length=(max_len,), mask_zero=False)(
        inp1)  # 20-dim embedding
    # emb3 = Embedding(input_dim=n_words + 1, output_dim=500, input_length=(max_len,), mask_zero=False)(
    #    inp2)  # 20-dim embedding
    # emb4 = Embedding(input_dim=n_words + 1, output_dim=50, input_length=(max_len,), mask_zero=False)(
    #    inp3)  # 20-dim embedding
    combined = concatenate([emb1, emb2])
    model = Bidirectional(LSTM(units=250, input_shape=emb1.shape, return_sequences=True,
                               recurrent_dropout=0.1), input_shape=emb1.shape)(combined)  # variational biLSTM
    model = TimeDistributed(Dense(250, activation="relu"))(model)  # a dense layer as suggested by neuralNer
    crf = CRF(10, sparse_target=False)  # CRF layer
    out = crf(model)
    model = Model(inputs=[inp0,inp1], outputs=out)
    model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
    model.load_weights("models_and_idxs/" + filename + ".nn")
    #save_load_utils.load_all_weights(model, "models_and_idxs/" + filename + ".nn")

    # load the dictionary of indexes

    return model, idx


############### output_entities function ###############
def output_entities(dataset, predictions, outfile):
    """
    Task : Output detected entities in the format expected by the evaluator

    Input :
    dataset : A dataset produced by load_data .
    predictions : For each sentence in dataset , a list with the labels for each
    sentence token, as predicted by the model

    Output :
    prints the detected entities to stdout in the format required by the
    evaluator.

    Example :
    >> output_entities(dataset, predictions)
    DDI - DrugBank.d283.s4 |14-35| bile acid sequestrants | group
    DDI - DrugBank.d283.s4 |99-104| tricor | group
    DDI - DrugBank.d283.s5 |22-33| cyclosporine | drug
    DDI - DrugBank.d283.s5 |196-208| fibrate drugs | group
    ...
    """
    #TODO: Poner bien las labels y juntar las palabras que sean B-algo I-algo
    beginningbegined = False
    actual_word = ""
    typeofword = ""
    with open(outfile, 'w') as output:
        for index_sid, sid in enumerate(dataset.keys()):
            for index_word in range(len(dataset[sid])):
                print(index_sid)
                print(index_word)
                if index_word == 100:
                    break
                if predictions[index_sid][index_word][0] != "O":
                    if predictions[index_sid][index_word][0] == "B-drug":
                        if beginningbegined:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword,
                                  file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug"
                        else:
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug"
                    elif predictions[index_sid][index_word][0] == "B-group":
                        if beginningbegined:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "group"
                        else:
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "group"
                    elif predictions[index_sid][index_word][0] == "B-brand":
                        if beginningbegined:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            ending_offset = dataset[sid][index_word][2]
                            starting_offset = dataset[sid][index_word][1]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "brand"
                        else:
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "brand"
                    elif predictions[index_sid][index_word][0] == "B-drug_n":
                        if beginningbegined:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug_n"
                        else:
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug_n"
                    elif predictions[index_sid][index_word][0] == "I-drug":
                        if typeofword=="drug":
                            actual_word = actual_word +" "+dataset[sid][index_word][0]
                            ending_offset = dataset[sid][index_word][2]
                        else:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug"
                    elif predictions[index_sid][index_word][0] == "I-group":
                        if typeofword == "group":
                            actual_word = actual_word + " " + dataset[sid][index_word][0]
                            ending_offset = dataset[sid][index_word][2]
                        else:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "group"
                    elif predictions[index_sid][index_word][0] == "I-brand":
                        if typeofword == "brand":
                            actual_word = actual_word + " " + dataset[sid][index_word][0]
                            ending_offset = dataset[sid][index_word][2]
                        else:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            starting_offset = dataset[sid][index_word][1]
                            ending_offset = dataset[sid][index_word][2]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "brand"
                    elif predictions[index_sid][index_word][0] == "I-drug_n":
                        if typeofword == "drug_n":
                            actual_word = actual_word + " " + dataset[sid][index_word][0]
                            ending_offset = dataset[sid][index_word][2]
                        else:
                            print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                                  "|" + actual_word + "|" + typeofword, file=output)
                            beginningbegined = True
                            ending_offset = dataset[sid][index_word][2]
                            starting_offset = dataset[sid][index_word][1]
                            actual_word = dataset[sid][index_word][0]
                            typeofword = "drug_n"
                elif predictions[index_sid][index_word][0] == "O" or predictions[index_sid][index_word][0] == "<PAD>":
                    if beginningbegined:
                        print(sid + "|" + str(starting_offset) + "-" + str(ending_offset) +
                              "|" + actual_word + "|" + typeofword, file=output)
                        beginningbegined = False
