
import re
import string

import numpy as np

np.random.seed(1337) # for reproducibility
from keras.layers import Conv1D, MaxPooling1D, Embedding, SpatialDropout1D
from keras.layers import LSTM

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences


train_data = pd.read_pickle('./subset')

print("Summary of groups")
print(train_data.groupby('category').count())

docs_global = train_data['text_cleaned']

# create the tokenizer
t = Tokenizer()
t.fit_on_texts(docs_global)


# summarize what was learned
print('Test plus Train data documents:')
print(t.document_count)


# Encode Train data
encoded_seqs = t.texts_to_sequences(docs_global)


# Pad Train Data
encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

# Scaler
# scaler = StandardScaler().fit(encoded_seqs_pad)
# encoded_seqs_pad_scaled = scaler.transform(encoded_seqs_pad)
encoded_seqs_pad_scaled = encoded_seqs_pad

print("Encoded Seqs Shape")
print(encoded_seqs_pad.shape)

# categories - sklearn
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(train_data["category"])


print('X Data Size')
print(encoded_seqs_pad_scaled.shape)
print('Y Data Size')
print(lb_results.shape)

shape_encoded = encoded_seqs_pad_scaled.shape
# all data (train)
Y_pre = lb_results
X_pre = encoded_seqs_pad_scaled
# X_pre = encoded_seqs_pad_scaled.reshape([shape_encoded[0], shape_encoded[1], 1])


# Split Data
x_train, x_val, y_train, y_val = train_test_split(X_pre, Y_pre, test_size=0.3)


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())

score_all = model.evaluate(X_pre, Y_pre, verbose=0)
score_all_test = model.evaluate(x_val, y_val, verbose=0)
print('All data acc: ' + str(score_all[1]))
print('Val data acc: ' + str(score_all_test[1]))

counter = 1
while score_all[1] <= 0.9:
    print('counter steps: ' + str(counter))
    model.fit(x_train, y_train,
                         validation_data=[x_val, y_val],
                         epochs=1,
                         batch_size=8,
                         verbose=2)

    score_all = model.evaluate(X_pre, Y_pre, verbose=2)
    score_all_test = model.evaluate(x_val, y_val, verbose=0)
    print('All data acc: ' + str(score_all[1]))
    print('Val data acc: ' + str(score_all_test[1]))
    print('--------------')
    counter += 1


pred = model.predict(X_pre)
df_pred = pd.DataFrame(argmax(pred, 1))
df_pred['real'] = argmax(Y_pre, 1)


pred = model.predict(x_val)
df_pred = pd.DataFrame(argmax(pred, 1))
df_pred['real'] = argmax(y_val, 1)


