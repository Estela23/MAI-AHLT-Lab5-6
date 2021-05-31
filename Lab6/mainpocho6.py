import pickle

import numpy as np

from Lab6.cosaspochas import learnpocho
from Lab6.utils import load_data, create_indexes, encode_words, encode_labels, load_model_and_indexes, \
    output_interactions
from learner import learn
from classifier import predict
import sys
import tensorflow as tf


encodear = int(sys.argv[1])


train_dir = "../data/train/"
validation_dir = "../data/devel/"
test_dir = "../data/test/"

model_name = "first_try_DDI"    # Update each time the name of the model to a more explicative name

outfile = "results/output-first_try_DDI4.txt"
if encodear==1:
    train_data = load_data(train_dir)
    val_data= load_data(validation_dir)

    max_len = 165    # 200
    # create indexes from training data
    idx = create_indexes(train_data, max_len)
    pickle.dump(idx, open("Lab6/modelidx.idx", 'wb'))

    # encode datasets
    X_train = encode_words(train_data, idx)
    pickle.dump(X_train, open("Lab6/DataTrain.data", 'wb'))
    Y_train = encode_labels(train_data, idx)
    pickle.dump(Y_train, open("Lab6/LabelTrain.data", 'wb'))
    X_val = encode_words(val_data, idx)
    pickle.dump(X_val, open("Lab6/DataVal.data", 'wb'))
    Y_val = encode_labels(val_data, idx)
    pickle.dump(Y_val, open("Lab6/LabelVal.data", 'wb'))

    test_data = load_data(test_dir)
    X = encode_words(test_data, idx)
    pickle.dump(X, open("Lab6/DataTest.data", 'wb'))
idx = pickle.load(open("Lab6/modelidx.idx", 'rb'))
X_train = pickle.load(open("Lab6/DataTrain.data", 'rb'))
X_val = pickle.load(open("Lab6/DataVal.data", 'rb'))
Y_train = pickle.load(open("Lab6/LabelTrain.data", 'rb'))
Y_val = pickle.load(open("Lab6/LabelVal.data", 'rb'))
test_data = pickle.load(open("Lab6/DataTest.data", 'rb'))
# Parse data in the xml files and train a model
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))
# Parse data in the xml files and train a model
model = learnpocho(np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val), idx, model_name)

# Predict with the trained model about the data in test_dir
#predict(model_name, test_dir, outfile)
#model2, idx = load_model_and_indexes(model_name)
# load data to annotate
#test_data = load_data(data_dir)

# encode dataset
#X = encode_words(test_data, idx)
#pickle.dump(X, open("Lab6/DataTest.data", 'wb'))
inp1 = np.array([[item[0] for item in sublist] for sublist in test_data])
inp2 = np.array([[item[1] for item in sublist] for sublist in test_data])
inp3 = np.array([[item[2] for item in sublist] for sublist in test_data])
# tag sentences in dataset
Y = model.predict([inp1,inp2,inp3])
# get most likely tag for each pair
Y = [list(idx['labels'].keys())[np.argmax(y)] for y in Y]
#Y = [[[key for (key, value) in idx['labels'].items() if value == np.argmax(y)] for y in s] for s in Y]
test_data = load_data(test_dir)
# extract entities and dump them to output file
output_interactions(test_data, Y, outfile)
