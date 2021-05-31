import numpy as np
import sklearn.preprocessing

from utils import load_data, create_indexes, encode_words, encode_labels, save_model_and_indexes, \
    build_network  # build_network,


def learnpocho(X_train, Y_train, X_val, Y_val, idx, model_name):
    """
    learns a NN model using train_dir as training data , and validation_dir
    as validation data . Saves learnt model in a file named model_name
    """

    # TODO:
    # build network
    model = build_network(idx)


    # TODO:
    # train model
    inp1 = np.array([[item[0]  for item in sublist] for sublist in X_train])
    inp2 = np.array([[item[1] for item in sublist] for sublist in X_train])
    inp3 = np.array([[item[2] for item in sublist] for sublist in X_train])
    inp4 = np.array([[item[3] for item in sublist] for sublist in X_train])
    val1 = np.array([[item[0] for item in sublist] for sublist in X_val])
    val2 = np.array([[item[1] for item in sublist] for sublist in X_val])
    val3 = np.array([[item[2] for item in sublist] for sublist in X_val])
    val4 = np.array([[item[3] for item in sublist] for sublist in X_val])
    Y_train = np.array([[[0.0 if value!= item[0] else 1.0 for value in range(len(np.zeros((10,))))] for item in sublist] for sublist in Y_train])
    Y_val = np.array([[[0.0 if value!= item[0] else 1.0 for value in range(len(np.zeros((10,))))] for item in sublist] for sublist in Y_val])
    print(Y_train)
    model.fit([inp1,inp2,inp3,inp4], Y_train, validation_data=([val1,val2,val3,val4], Y_val), batch_size = 32, epochs= 1)

    # save model and indexes, for later use in prediction
    save_model_and_indexes(model, idx, model_name)
    return model
