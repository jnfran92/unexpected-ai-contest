import os
import pickle
import time

import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Activation, GlobalMaxPooling1D, Flatten
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import pandas as pd
from numpy import argmax

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

batches_path = './train_data/batches/spanish'
n_batch = 0  # Batch to train

model_name = "spanish_cnn_dense_2_b_"

print("Reading train batch and val SPANISH")
print("Reading validation")
x_val0 = np.load(batches_path + '/' + 'x_val' + str(0) + '.npy')
# x_val1 = np.load(batches_path + '/' + 'x_val' + str(1) + '.npy')

y_val0 = np.load(batches_path + '/' + 'y_val' + str(0) + '.npy')
# y_val1 = np.load(batches_path + '/' + 'y_val' + str(1) + '.npy')

x_val = x_val0
y_val = y_val0

print("Reading batch")
start = time.time()
x_train = np.load(batches_path + '/' + 'x_train_' + str(n_batch) + '.npy')
y_train = np.load(batches_path + '/' + 'y_train_' + str(n_batch) + '.npy')
stop = time.time()
print(stop - start)

print("Reading Tokenizer")
t = Tokenizer()
print("Loading the tokenizer")
with open('./tokenizer/tokenizer_spanish.pickle', 'rb') as handle:
    t = pickle.load(handle)

max_num_words_vocabulary = len(t.index_word)
print("Max num words vocabulary: " + str(max_num_words_vocabulary))
del t

print("Creating Model")
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = max_num_words_vocabulary
# This is fixed.
EMBEDDING_DIM = int(MAX_NB_WORDS*(5/5))

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(filters=512, kernel_size=2, padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=4, padding='same', activation='relu'))
model.add(Conv1D(filters=512, kernel_size=8, padding='same', activation='relu'))
model.add(GlobalMaxPooling1D())

model.add(Dense(2048))
model.add(Dense(2048))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(y_train.shape[1], activation='softmax'))
# adam_opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Create Callback
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=1)

csv_logger = CSVLogger(filename="./logs/" + model_name + str(n_batch) + ".csv")

# Train
fit_data = model.fit(x_train, y_train,
                     validation_data=[x_val, y_val],
                     epochs=3,
                     batch_size=2048,
                     verbose=2,
                     callbacks=[early_stop, csv_logger])

#
# print('Predicting data-------')
# pred = model.predict(x_val)
# df_pred = pd.DataFrame(argmax(pred, 1))
# df_pred['real'] = argmax(y_val, 1)
# print('Prediction on Val errors: ' + str(sum(df_pred[0] != df_pred['real'])))


# Save model
print('Saving the Model')
model_json = model.to_json()
file_model_name = model_name + str(n_batch)
with open('./models/spanish/' + file_model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./models/spanish/' + file_model_name + '.h5')

