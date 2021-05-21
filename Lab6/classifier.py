from utils import load_model_and_indexes, load_data, encode_words, output_interactions
import numpy as np


def predict(model_name, data_dir, outfile):
    """
    Loads a NN model from file 'model_name' and uses it to extract drugs
    in data_dir . Saves results to 'outfile' in the appropriate format .
    """

    # load model and associated encoding data
    model, idx = load_model_and_indexes(model_name)
    # load data to annotate
    test_data = load_data(data_dir)

    # encode dataset
    X = encode_words(test_data, idx)

    # tag sentences in dataset
    Y = model.predict(X)
    # get most likely tag for each pair
    Y = [idx['labels'][np.argmax(y)] for y in Y]

    # extract entities and dump them to output file
    output_interactions(test_data, Y, outfile)

    # evaluate using official evaluator
    # evaluation(data_dir, outfile)
