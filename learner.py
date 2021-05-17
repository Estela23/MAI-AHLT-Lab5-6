from utils import load_data, create_indexes, build_network, encode_words, encode_labels, save_model_and_indexs


def learn(train_dir, validation_dir, model_name):
    """
    learns a NN model using train_dir as training data , and validation_dir
    as validation data . Saves learnt model in a file named model_name
    """
    # load train and validation data in a suitable form
    train_data = load_data(train_dir)
    val_data = load_data(validation_dir)

    # create indexes from training data
    max_len = 100
    idx = create_indexes(train_data, max_len)

    # build network
    model = build_network(idx)

    # encode datasets
    X_train = encode_words(train_data, idx)
    Y_train = encode_labels(train_data, idx)
    X_val = encode_words(val_data, idx)
    Y_val = encode_labels(val_data, idx)

    # train model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val))

    # save model and indexes, for later use in prediction
    save_model_and_indexs(model, idx, model_name)
