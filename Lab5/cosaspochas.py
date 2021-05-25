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
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size = 32, epochs= 4)

    # save model and indexes, for later use in prediction
    save_model_and_indexes(model, idx, model_name)
    return model
