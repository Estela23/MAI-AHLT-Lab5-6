from utils import load_data, create_indexes, encode_words, encode_labels, save_model_and_indexes    # build_network,


def learn(train_dir, validation_dir, model_name):
    """
    learns a NN model using train_dir as training data , and validation_dir
    as validation data . Saves learnt model in a file named model_name
    """
    # load train and validation data in a suitable form
    train_data = load_data(train_dir)
    val_data = load_data(validation_dir)

    # the maximum length of the sentences in the train data is 165, however there is only 7 sentences with lengths > 100
    max_len = 100
    # create indexes from training data
    idx = create_indexes(train_data, max_len)

    # TODO:
    # build network
    # model = build_network(idx)

    # encode datasets
    X_train = encode_words(train_data, idx)
    Y_train = encode_labels(train_data, idx)
    X_val = encode_words(val_data, idx)
    Y_val = encode_labels(val_data, idx)

    # TODO:
    # train model
    # model.fit(X_train, Y_train, validation_data=(X_val, Y_val))

    # save model and indexes, for later use in prediction
    # save_model_and_indexes(model, idx, model_name)
