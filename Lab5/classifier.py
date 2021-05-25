from utils import load_model_and_indexes, load_data, encode_words, output_entities
import numpy as np


def predict(model_name, data_dir, outfile, idx, X_train, Y_train, X_val, Y_val):
    """
    Loads a NN model from file 'model_name' and uses it to extract drugs
    in data_dir . Saves results to 'outfile' in the appropriate format .
    """

    # load model and associated encoding data
    model, idx = load_model_and_indexes(model_name, idx, X_train, Y_train, X_val, Y_val)
    # load data to annotate
    test_data, _ = load_data(data_dir)

    # encode dataset
    X = encode_words(test_data, idx)

    # tag sentences in dataset
    Y = model.predict(np.array(X))
    # get most likely tag for each word
    Y = [[idx['labels'][np.argmax(y)] for y in s] for s in Y]

    # extract entities and dump them to output file
    output_entities(test_data, Y, outfile)

    # evaluate using official evaluator .
    # evaluation(data_dir, outfile)
