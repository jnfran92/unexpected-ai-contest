import os
import pickle
import time

import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import pandas as pd
from numpy import argmax
from keras.models import model_from_json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1337)  # for reproducibility

batches_path = './train_data/batches/spanish'
n_batch = 1  # Batch to train

file_model_name = "spanish_cnn_1_32_lstm_120_b_" + str(n_batch)

print("Reading train batch and val SPANISH")
print("Reading validation")
x_val0 = np.load(batches_path + '/' + 'x_val' + str(0) + '.npy')
x_val1 = np.load(batches_path + '/' + 'x_val' + str(1) + '.npy')

y_val0 = np.load(batches_path + '/' + 'y_val' + str(0) + '.npy')
y_val1 = np.load(batches_path + '/' + 'y_val' + str(1) + '.npy')

x_val = np.concatenate((x_val0, x_val1))
y_val = np.concatenate((y_val0, y_val1))

print("Reading batch")
start = time.time()
x_train = np.load(batches_path + '/' + 'x_train_' + str(n_batch) + '.npy')
y_train = np.load(batches_path + '/' + 'y_train_' + str(n_batch) + '.npy')
stop = time.time()
print(stop - start)
#
# print("Reading Tokenizer")
# t = Tokenizer()
# print("Loading the tokenizer")
# with open('./tokenizer/tokenizer_spanish.pickle', 'rb') as handle:
#     t = pickle.load(handle)
#
# max_num_words_vocabulary = len(t.index_word)
# print("Max num words vocabulary: " + str(max_num_words_vocabulary))
# del t

print("Loading Model")
model_name = "spanish_model_fa"
# load json and create model
json_file = open('./models/spanish/' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/spanish/' + model_name + ".h5")
print("Loaded model from disk")

# Common operation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Create Callback
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=3,
                           verbose=1)

csv_logger = CSVLogger(filename="./logs/" + file_model_name + ".csv")

# Train
fit_data = model.fit(x_train, y_train,
                     validation_data=[x_val, y_val],
                     epochs=15,
                     batch_size=128,
                     verbose=2,
                     callbacks=[early_stop, csv_logger])


print('Predicting data-------')
pred = model.predict(x_val)
df_pred = pd.DataFrame(argmax(pred, 1))
df_pred['real'] = argmax(y_val, 1)
print('Prediction on Val errors: ' + str(sum(df_pred[0] != df_pred['real'])))


# Save model
print('Saving the Model')
model_json = model.to_json()
# file_model_name = "spanish_cnn_1_32_lstm_120_b_" + str(n_batch)
with open('./models/spanish/' + file_model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./models/spanish/' + file_model_name + '.h5')

