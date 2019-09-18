
import os
import pickle
import time

import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from numpy import argmax
from keras.models import model_from_json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1337)  # for reproducibility

# Read Test
test_data = pd.read_pickle('./data/test_subset_spanish.pkl').reset_index(drop=True)


print("Reading Tokenizer")
t = Tokenizer()
print("Loading the tokenizer")
with open('./tokenizer/tokenizer_spanish.pickle', 'rb') as handle:
    t = pickle.load(handle)


print("Reading Binarizer")
lb_style = LabelBinarizer()
with open('./binarizer/binarizer_spanish.pickle', 'rb') as handle:
    lb_style = pickle.load(handle)


# Loading Model
n_batch = 3
print("Loading Model")
model_name = "spanish_cnn_32_64_lstm_200_b_" + str(n_batch)
print("Loading model: " + model_name)
# load json and create model
json_file = open('./models/spanish/' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/spanish/' + model_name + ".h5")
print("Loaded model from disk")
print(model.summary())

# Creating test data set
print("Creating test data set")
encoded_seqs = t.texts_to_sequences(test_data['text_cleaned'])

# Pad Test Data
print("Pad Test Data")
max_len_encoded = model.input_shape[1]
encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=max_len_encoded, dtype='int32', padding='pre', truncating='pre', value=0.0)


# Def function to get labels
def get_label_name(index):
    return lb_style.classes_[index]


# Predicting test data
pred = model.predict(encoded_seqs_pad)

df_pred = pd.DataFrame(argmax(pred, 1))
out_data = pd.DataFrame(df_pred[0].map(get_label_name))
out_data.columns = ['category']
out_data.index.name = 'id'

out_data['title'] = test_data['text_cleaned']

out_data.to_csv("./results.csv", header=True)




#
# batches_path = './train_data/batches/spanish'
# n_batch = 1  # Batch to train
#
# file_model_name = "spanish_cnn_32_64_lstm_200_b_" + str(n_batch)
#
# print("Reading train batch and val SPANISH")
# print("Reading validation")
# x_val0 = np.load(batches_path + '/' + 'x_val' + str(0) + '.npy')
# x_val1 = np.load(batches_path + '/' + 'x_val' + str(1) + '.npy')
#
# y_val0 = np.load(batches_path + '/' + 'y_val' + str(0) + '.npy')
# y_val1 = np.load(batches_path + '/' + 'y_val' + str(1) + '.npy')
#
# x_val = np.concatenate((x_val0, x_val1))
# y_val = np.concatenate((y_val0, y_val1))
#
# print("Reading batch")
# start = time.time()
# x_train = np.load(batches_path + '/' + 'x_train_' + str(n_batch) + '.npy')
# y_train = np.load(batches_path + '/' + 'y_train_' + str(n_batch) + '.npy')
# stop = time.time()
# print(stop - start)








#
# # Common operation
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# # Create Callback
# early_stop = EarlyStopping(monitor='val_loss',
#                            min_delta=0,
#                            patience=4,
#                            verbose=1)
#
# csv_logger = CSVLogger(filename="./logs/" + file_model_name + ".csv")
#
# # Train
# fit_data = model.fit(x_train, y_train,
#                      validation_data=[x_val, y_val],
#                      epochs=20,
#                      batch_size=128,
#                      verbose=2,
#                      callbacks=[early_stop, csv_logger])
#
#
# print('Predicting data-------')
# pred = model.predict(x_val)
# df_pred = pd.DataFrame(argmax(pred, 1))
# df_pred['real'] = argmax(y_val, 1)
# print('Prediction on Val errors: ' + str(sum(df_pred[0] != df_pred['real'])))

#
# # Save model
# print('Saving the Model')
# model_json = model.to_json()
# # file_model_name = "spanish_cnn_1_32_lstm_120_b_" + str(n_batch)
# with open('./models/spanish/' + file_model_name + '.json', "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights('./models/spanish/' + file_model_name + '.h5')
#
