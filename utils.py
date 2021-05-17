def load_data(data_dir) :
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
    {'DDI - DrugBank . d370 .s0 ': [(' as ',0,1,'O '), (' differin ',3,10,'B- brand '),
    (' gel ',12,14,'O '), ... , (' with ' ,343 ,346 , 'O '),
    (' caution ' ,348 ,354 , 'O '), ( '. ' ,355 ,355 , 'O ')],
    'DDI - DrugBank . d370 .s1 ': [(' particular ',0,9,'O '), (' caution ',11,17,'O '),
    (' should ',19,24,'O '), ... ,( ' differin ' ,130 ,137 , 'B- brand '),
    (' gel ',139, 141 ,'O '), ( '. ' ,142 ,142 , 'O ')],
    ...
    }
    """


def create_indexes(dataset, max_length):
    """
    Task :
    Create index dictionaries both for input ( words ) and output ( labels )
    from given dataset .
    Input :
    dataset : dataset produced by load_data .
    max_length : maximum length of a sentence ( longer sentences will
    be cut , shorter ones will be padded ).
    Output :
    A dictionary where each key is an index name (e.g. " words ", " labels ") ,
    and the value is a dictionary mapping each word / label to a number .
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


def build_network(idx) :
    """
    Task : Create network for the learner.
    Input :
    idx : index dictionary with word/labels codes, plus maximum sentence length.
    Output :
    Returns a compiled Keras neural network with the specified layers
    """
    # sizes
    n_words = len(idx['words'])
    n_labels = len(idx['labels'])
    max_len = idx['max_len']

    # create network layers
    inp = Input(shape=(max_len,))
    ## ... add missing layers here ... #
    out = # final output layer

    # create and compile model
    model = Model(inp , out )
    model.compile() # set appropriate parameters ( optimizer , loss , etc )

    return model


def encode_words(dataset, idx):
    """
    Task :
    Encode the words in a sentence dataset formed by lists of tokens into
    lists of indexes suitable for NN input.

    Input :
    dataset : A dataset produced by load_data.
    idx : A dictionary produced by create_indexes , containing word and
    label indexes, as well as the maximum sentence length.

    Output :
    The dataset encoded as a list of sentence, each of them is a list of
    word indices. If the word is not in the index, <UNK > code is used. If
    the sentence is shorter than max_len it is padded with <PAD > code.

    Example :
    >> encode_words ( train_data , idx )
    [ [6882 1049 4911 ... 0 0 0 ]
    [2290 7548 8069 ... 0 0 0 ]
    ...
    [2002 6582 7518 ... 0 0 0 ] ]
    """


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
    padded with <PAD > code .

    Example :
    >> encode_labels(train_data, idx)
    [[ [4] [6] [4] [4] [4] [4] ... [0] [0] ]
    [ [4] [4] [8] [4] [6] [4] ... [0] [0] ]
    ...
    [ [4] [8] [9] [4] [4] [4] ... [0] [0] ]
    ]
    """


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


def load_model_and_indexes(filename):
    """
    Task : Load model and associate indexes from disk
    Input :
    filename : filename to be loaded
    Output :
    Loads a model from filename .nn, and its indexes from filename .idx
    Returns the loaded model and indexes .
    """


def output_entities(dataset, preds):
    """
    Task : Output detected entities in the format expected by the evaluator

    Input :
    dataset : A dataset produced by load_data .
    preds : For each sentence in dataset , a list with the labels for each
    sentence token, as predicted by the model

    Output :
    prints the detected entities to stdout in the format required by the
    evaluator .

    Example :
    >> output_entities(dataset, preds)
    DDI - DrugBank . d283 .s4 |14 -35| bile acid sequestrants | group
    DDI - DrugBank . d283 .s4 |99 -104| tricor | group
    DDI - DrugBank . d283 .s5 |22 -33| cyclosporine | drug
    DDI - DrugBank . d283 .s5 |196 -208| fibrate drugs | group
    ...
    """
