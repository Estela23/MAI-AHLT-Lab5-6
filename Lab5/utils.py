from nltk.tokenize import TreebankWordTokenizer as twt
from os import listdir
from xml.dom.minidom import parse
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import pickle


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
                    # see if the  token  is part of an entity , and  which  part (B/I)
                    tag = get_tag(tokens[i], gold)
                    tokens[i] = tokens[i] + (tag,)
                parsed_data[sid] = tokens
    return parsed_data, max_length


############### create_indexes function ###############
def create_indexes(dataset, max_length):
    """
    Task :
    Create index dictionaries both for input ( words ) and output ( labels )
    from given dataset .
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
                if token[0][-5:].lower() not in suffixes:
                    suffixes[token[0][-5:].lower()] = idx_suffixes
                    idx_suffixes += 1
                if token[3] not in labels:
                    labels[token[3]] = idx_labels
                    idx_labels += 1
    # Return the definitive dictionary with all the information retrieved
    return {"words": words, "lemmas": lemmas, "pos": pos, "suffixes": suffixes, "labels": labels, "max_len": max_length}


############### build_network function ###############
"""def build_network(idx):
    ""
    Task : Create network for the learner.
    Input :
    idx : index dictionary with word/labels codes, plus maximum sentence length.
    Output :
    Returns a compiled Keras neural network with the specified layers
    ""
    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['max_len']

    # create network layers
    inp = Input(shape=(max_len,))
    ## ... add missing layers here ... #
    out = # final output layer

    # create and compile model
    model = Model(inp, out)
    model.compile() # set appropriate parameters (optimizer, loss, etc)

    return model"""


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
    The dataset encoded as a list of sentence, each of them is a list of
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
            this_words = [idx["words"][word[0].lower()] if word[0].lower() in idx["words"] else idx["words"]["<UNK>"] for word in sentence]
            this_lemmas = [idx["lemmas"][lemma] if lemma in idx["lemmas"] else idx["lemmas"]["<UNK>"] for lemma in sentence_lemmas]
            this_pos = [idx["pos"][pos] if pos in idx["pos"] else idx["pos"]["<UNK>"] for pos in sentence_pos]
            this_suffixes = [idx["suffixes"][word[0][-5:].lower()] if word[0][-5:].lower() in idx["suffixes"] else idx["suffixes"]["<UNK>"] for word in sentence]

            this_padding = [idx["words"]["<PAD>"] for _ in range(idx["max_len"] - len(sentence))]
            this_words.extend(this_padding)
            this_lemmas.extend(this_padding)
            this_pos.extend(this_padding)
            this_suffixes.extend(this_padding)

            this_sentence_info = [list(elem) for elem in zip(this_words, this_lemmas, this_pos, this_suffixes)]
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
    # save the model
    pickle.dump(model, open(filename + ".nn", 'wb'))

    # save the dictionary of indexes
    pickle.dump(idx, open(filename + ".idx", 'wb'))


############### load_model_and_indexes function ###############
def load_model_and_indexes(filename):
    """
    Task : Load model and associate indexes from disk
    Input :
    filename : filename to be loaded
    Output :
    Loads a model from filename .nn, and its indexes from filename .idx
    Returns the loaded model and indexes.
    """

    # load the model
    model = pickle.load(open(filename + ".nn", 'rb'))

    # load the dictionary of indexes
    idx = pickle.load(open(filename + ".idx", 'rb'))

    return model, idx


############### output_entities function ###############
def output_entities(dataset, preds, outfile):
    """
    Task : Output detected entities in the format expected by the evaluator

    Input :
    dataset : A dataset produced by load_data .
    preds : For each sentence in dataset , a list with the labels for each
    sentence token, as predicted by the model

    Output :
    prints the detected entities to stdout in the format required by the
    evaluator.

    Example :
    >> output_entities(dataset, preds)
    DDI - DrugBank . d283 .s4 |14 -35| bile acid sequestrants | group
    DDI - DrugBank . d283 .s4 |99 -104| tricor | group
    DDI - DrugBank . d283 .s5 |22 -33| cyclosporine | drug
    DDI - DrugBank . d283 .s5 |196 -208| fibrate drugs | group
    ...
    """
