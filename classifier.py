from utils import load_model_and_indexs, load_data, encode_words, encode_labels, output_entities
import numpy as np


def predict(model_name, data_dir, outfile):
    """
    Loads a NN model from file 'model_name ' and uses it to extract drugs
    in datadir . Saves results to 'outfile ' in the appropriate format .
    """

    # load model and associated encoding data
    model, idx = load_model_and_indexs(model_name)
    # load data to annotate
    test_data = load_data(data_dir)

    # encode dataset
    X = encode_words(test_data, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each word
    Y = [[idx['labels '][np.argmax(y)] for y in s] for s in Y]

    # extract entities and dump them to output file
    output_entities(test_data, Y, outfile)

    # evaluate using official evaluator .
    # evaluation(data_dir, outfile)
