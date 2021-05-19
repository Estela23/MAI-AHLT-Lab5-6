from utils import load_data, create_indexes, encode_words, encode_labels, save_model_and_indexes    # build_network,


def learn(train_dir, validation_dir, model_name):
    """
    learns a NN model using train_dir as training data , and validation_dir
    as validation data . Saves learnt model in a file named model_name
    """
    # load train and validation data in a suitable form
    train_data, max_length = load_data(train_dir)
    val_data, _ = load_data(validation_dir)

    # create indexes from training data
    max_len = 100   # TODO: cambiar esto a la máxima longitud del train? es 165
    idx = create_indexes(train_data, max_len)

    # TODO: al crear la lista de sufijos por ejemplo queremos pasar por alto la puntuación y tal o no?
    # TODO: Si la pasamos por alto luego en el test va a ser todo <UNK>... que será mejor?

    # TODO: este paso es tuyo Fer
    # build network
    # model = build_network(idx) TODO: uncomment

    # encode datasets
    X_train = encode_words(train_data, idx)
    Y_train = encode_labels(train_data, idx)
    X_val = encode_words(val_data, idx)
    Y_val = encode_labels(val_data, idx)

    # TODO: a partir de aquí todo tuyo Fer
    # train model
    # model.fit(X_train, Y_train, validation_data=(X_val, Y_val)) TODO: uncomment

    # save model and indexes, for later use in prediction
    # save_model_and_indexes(model, idx, model_name) TODO: uncomment
