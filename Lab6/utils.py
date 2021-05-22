from nltk.tokenize import TreebankWordTokenizer as twt
from os import listdir
from xml.dom.minidom import parse
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import itertools
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
    # TODO: if we want to consider only lowercase words meterle aquí un .lower antes del "for"
    tokens = [s[list_offset[i][0]:list_offset[i][1]] for i in range(len(list_offset))]
    start_tokens = [list_offset[i][0] for i in range(len(list_offset))]
    return tokens, start_tokens


# TODO; Masking the target drugs as e.g. <DRUG1>, <DRUG2>, and the rest as <DRUG <OTHER> will help the algorithm
#       generalize and avoid it focusing in the drug names, which are not relevant for the DDI task
#       (and also make it easier for it to spot the target entities).
def load_data(data_dir):
    """
    Task :
    Load XML files in given directory, tokenize each sentence, and extract
    learning examples ( tokenized sentence + entity pair)
    Input :
    data_dir : A directory containing XML files.
    Output :
    A list of classification cases. Each case is a list containing sentence
    id, entity1 id, entity2 id, ground truth relation label, and a list
    of sentence tokens (each token containing any needed information: word,
    lemma, PoS, offsets, etc)

    Example
        >> load_data(’data/Train’)
        [[’DDI-DrugBank.d66.s0’, ’DDI-DrugBank.d66.s0.e0’, ’DDI-DrugBank.d66.s0.e1’, ’null’,
          [(’<DRUG1>’, ’<DRUG1>’, ’<DRUG1>’), (’-’, ’-’, ’:’), (’Concomitant’, ’concomitant’, ’JJ’),
           (’use’, ’use’, ’NN’), (’of’, ’of’, ’IN’), (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’),
           (’and’, ’and’, ’CC’), (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’), (’may’, ’may’, ’MD’),
           ..., (’syndrome’, ’syndrome’, ’NN’), (’.’ ,’.’, ’.’)]
         ]
        ...
         [’DDI-MedLine.d94.s12’, ’DDI-MedLine.d94.s12 . e1 ’, ’DDI - MedLine . d94 . s12 . e2 ’, ’ effect ’,
          [(’The’, ’the’, ’DT’), (’uptake’, ’uptake’, ’NN’), (’inhibitors’, ’inhibitor’, ’NNS’),
           (’<DRUG_OTHER>’, ’<DRUG_OTHER>’, ’<DRUG_OTHER>’), (’and’, ’and’, ’CC’), (’<DRUG1>’, ’<DRUG1>’, ’<DRUG1>’),
            ...(’effects’, ’effect’, ’NNS’), (’of’, ’of’, ’IN’), (’<DRUG2>’, ’<DRUG2>’, ’<DRUG2>’),
            (’in’, ’in’, ’IN’), ...]
         ]
        ...
        ]
    """

    lemmatizer = WordNetLemmatizer()

    # Initialize list to return parsed data
    parsed_data = []

    # process each file in directory
    for f in listdir(data_dir):
        # parse XML file, obtaining a DOM tree
        tree = parse(data_dir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value  # get sentence id
            s_text = s.attributes["text"].value  # get sentence text

            # tokenize text
            tokens, start_tokens = tokenize(s_text)

            s_lemmas = [lemmatizer.lemmatize(token.lower()) for token in tokens]
            s_pos = [pos_tag(tokens)[i][1] for i in range(len(tokens))]

            this_sentence_tuples = [tuple(elem) for elem in zip(start_tokens, tokens, s_lemmas, s_pos)]

            # load sentence entities into a dictionary
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e.attributes["id"].value
                entities[eid] = e.attributes["charOffset"].value.split("-")

            starts_entities = [entities[key][0] for key in entities.keys()]

            # TODO: aqui estan guardados los start-offset de las entities, con un itertools combinations haré
            #       todos los pares de entities posibles, para en cada idx_pair, p in enumerate(pairs) substituir
            #       las entities por (<DRUG1>, <DRUG1>, <DRUG1>, <DRUG1>) etc.
            # TODO: ojo, quitar el Star-offset de todas las palabras después!

            pairs_indexes = [x for x in itertools.combinations([i for i in range(len(starts_entities))], 2)]

            if s_text != '':
                # for each pair in the sentence add it to the list of parsed data
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    # get ground truth
                    ddi = p.attributes["ddi"].value
                    ddi_type = p.attributes["type"].value if ddi == "true" else "null"
                    # target entities
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    this_sentence_info = [sid, id_e1, id_e2, ddi_type, ]

    return parsed_data


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

    # Initialize dictionaries for each type of information with the corresponding indexes for Padding and Unknown values
    words = {"<PAD>": 0, "<UNK>": 1}
    idx_words = 2
    lemmas = {"<PAD>": 0, "<UNK>": 1}
    idx_lemmas = 2
    pos = {"<PAD>": 0, "<UNK>": 1}
    idx_pos = 2
    labels = {"<PAD>": 0}
    idx_labels = 1

    for sentence in dataset:
        if len(sentence[4]) < max_length:
            # Add elements to the dictionaries if they still do not exist
            for token in sentence[4]:
                if token[0].lower() not in words:
                    words[token[0].lower()] = idx_words
                    idx_words += 1
                if token[1] not in lemmas:
                    lemmas[token[1]] = idx_lemmas
                    idx_lemmas += 1
                if token[2] not in pos:
                    pos[token[2]] = idx_pos
                    idx_pos += 1
                if token[3] not in labels:
                    labels[token[3]] = idx_labels
                    idx_labels += 1
    # Return the definitive dictionary with all the information retrieved
    return {"words": words, "lemmas": lemmas, "pos": pos, "labels": labels, "max_len": max_length}


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
    n_lemmas = len(idx['lemmas'])
    n_pos = len(idx['pos'])
    n_suffixes = len(idx['suffixes'])
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
            this_words = [idx["words"][word[0].lower()] if word[0].lower() in idx["words"] else idx["words"]["<UNK>"] for word in sentence]
            this_lemmas = [idx["lemmas"][lemma] if lemma in idx["lemmas"] else idx["lemmas"]["<UNK>"] for lemma in sentence_lemmas]
            this_pos = [idx["pos"][pos] if pos in idx["pos"] else idx["pos"]["<UNK>"] for pos in sentence_pos]

            this_padding = [idx["words"]["<PAD>"] for _ in range(idx["max_len"] - len(sentence))]
            this_words.extend(this_padding)
            this_lemmas.extend(this_padding)
            this_pos.extend(this_padding)

            this_sentence_info = [list(elem) for elem in zip(this_words, this_lemmas, this_pos)]
            encoded_words.append(this_sentence_info)

    return encoded_words


############### encode_labels function ###############
# TODO; The shape of the produced list may need to be adjusted depending
#       on the architecture of your network and the kind of output layer you use.
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
    pickle.dump(model, open("Lab6/models_and_idxs/" + filename + ".nn", 'wb'))

    # save the dictionary of indexes
    pickle.dump(idx, open("Lab6/models_and_idxs/" + filename + ".idx", 'wb'))


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

    # load the model    # TODO: esto decía que lo hiciéramos con Keras, no idea "keras.models.load model"
    model = pickle.load(open("Lab6/models_and_idxs/" + filename + ".nn", 'rb'))

    # load the dictionary of indexes
    idx = pickle.load(open("Lab6/models_and_idxs/" + filename + ".idx", 'rb'))

    return model, idx


############### output_interactions function ###############
def output_interactions(dataset, predictions, outfile):
    """
    Task: Output detected DDIs in the format expected by the evaluator
    Input:
    dataset: A dataset produced by load_data.
    predictions: For each sentence in dataset, a label for its DDI type (or ’null’ if no DDI detected)
    Output:
    prints the detected interactions to stdout in the format required by the evaluator.
    Example:
    >> output_interactions(dataset, predictions)
    DDI - DrugBank.d398.s0 | DDI - DrugBank.d398.s0.e0 | DDI - DrugBank.d398.s0.e1 |
    effect
    DDI - DrugBank.d398.s0 | DDI - DrugBank.d398.s0.e0 | DDI - DrugBank.d398.s0.e2 |
    effect
    DDI - DrugBank.d211.s2 | DDI - DrugBank.d211.s2.e0 | DDI - DrugBank.d211.s2.e5 |
    mechanism
    ...
    """

    with open(outfile, 'w') as output:
        for index_pair in range(len(dataset)):
            if predictions[index_pair] != "null":
                print(dataset[index_pair][0] + "|" + dataset[index_pair][1] + "|" + dataset[index_pair][2] +
                      "|" + predictions[index_pair], file=output)
