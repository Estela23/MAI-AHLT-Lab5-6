import pickle

import numpy as np

from Lab5.cosaspochas import learnpocho
from Lab5.utils import load_data, create_indexes, encode_words, encode_labels, load_model_and_indexes, output_entities
from learner import learn
from classifier import predict
import sys
import tensorflow as tf
print(tf.__version__)
encodear = int(sys.argv[1])
train_dir = "../data/train/"
validation_dir = "../data/devel/"
test_dir = "../data/test/"
#TODO: guardar encodeds e idxs
model_name = "first_try_NER_PRUEBAS"    # Update each time the name of the model to a more explicative name

outfile = "results/output-first_try_NER_PRUEBAS.txt"
if encodear==1:
    train_data, max_length = load_data(train_dir)
    val_data, _ = load_data(validation_dir)

    max_len = max_length    # 200
    # create indexes from training data
    idx = create_indexes(train_data, max_len)
    pickle.dump(idx, open("Lab5/modelidx.idx", 'wb'))

    # encode datasets
    X_train = encode_words(train_data, idx)
    pickle.dump(X_train, open("Lab5/DataTrain.data", 'wb'))
    Y_train = encode_labels(train_data, idx)
    pickle.dump(Y_train, open("Lab5/LabelTrain.data", 'wb'))
    X_val = encode_words(val_data, idx)
    pickle.dump(X_val, open("Lab5/DataVal.data", 'wb'))
    Y_val = encode_labels(val_data, idx)
    pickle.dump(Y_val, open("Lab5/LabelVal.data", 'wb'))

    test_data, max_len_test = load_data(test_dir)
    print(max_len_test)
    pickle.dump(test_data, open("Lab5/DataTest.data", 'wb'))
    #TODO: guardarlo
idx = pickle.load(open("Lab5/modelidx.idx", 'rb'))
X_train = pickle.load(open("Lab5/DataTrain.data", 'rb'))
X_val = pickle.load(open("Lab5/DataVal.data", 'rb'))
Y_train = pickle.load(open("Lab5/LabelTrain.data", 'rb'))
Y_val = pickle.load(open("Lab5/LabelVal.data", 'rb'))
test_data = pickle.load(open("Lab5/DataTest.data", 'rb'))
# Parse data in the xml files and train a model
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))
model = learnpocho(np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), idx, model_name)#Cambiar a learn pocho

# Predict with the trained model about the data in test_dir
#predict(model_name, test_dir, outfile, idx, np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val))#Cambiar a predict pocho
#model, idx = load_model_and_indexes(model_name, idx, np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val))
# load data to annotate
#test_data, _ = load_data(test_dir)

# encode dataset
X = encode_words(test_data, idx)

# tag sentences in dataset
Y = model.predict(np.array(X))
# get most likely tag for each word
Y_aux=[]
for s in Y:
    to_append=[]
    for values in s:
        for i in range(len(values)):
            if values[i]==1:
                to_append.append(list(idx['labels'].keys())[i])
    Y_aux.append(to_append)
#Y = [[idx['labels'][np.argmax(y)] for y in s] for s in Y]

# extract entities and dump them to output file
output_entities(test_data, Y_aux, outfile)
