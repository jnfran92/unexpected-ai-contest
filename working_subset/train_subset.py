
import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

model_name = 'model_lstm_100_100'

train_data = pd.read_pickle('./train_subset')
test_data = pd.read_pickle('./test_subset')
print("Summary Train and test size: ")
print(train_data.shape)
print(test_data.shape)

print("Summary of groups")
print(train_data.groupby('category').count())

docs_global = pd.concat([train_data['text_cleaned'], test_data['text_cleaned']]).reset_index(drop=True)
print('Size of train + test: ' + str(docs_global.shape))
# docs_global = train_data['text_cleaned']

# create the tokenizer
t = Tokenizer()
t.fit_on_texts(docs_global)

# summarize what was learned
print('Test plus Train data documents:')
print(t.document_count)

t_index_words_len = t.index_word.__len__()
print('Vocabulary Tokenizer Size: ' + str(t_index_words_len))

# Encode Train data
encoded_seqs = t.texts_to_sequences(train_data['text_cleaned'])

# Pad Train Data
max_len_encoded = 25
encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=max_len_encoded, dtype='int32', padding='pre', truncating='pre', value=0.0)

# Scaler - There is no
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

t_index_words_len = t.index_word.__len__()
print('Vocabulary Tokenizer Size: ' + str(t_index_words_len))


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = t_index_words_len
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 210
# This is fixed.
EMBEDDING_DIM = 120

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, return_sequences=True))
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

score_all = model.evaluate(X_pre, Y_pre, verbose=0)
val_acc_temp = score_all[1]
# score_all_test = model.evaluate(x_val, y_val, verbose=0)
# print('All data acc: ' + str(score_all[1]))
print('Val data acc: ' + str(val_acc_temp))

counter = 1
max_counter = 8
while val_acc_temp <= 0.91:
    print('counter steps: ' + str(counter))
    fit_data = model.fit(x_train, y_train,
              validation_data=[x_val, y_val],
              epochs=1,
              batch_size=128,
              verbose=2)

    val_acc_temp = fit_data.history['val_acc'][0]

    if counter >= max_counter:
        break

    print('--------------')
    counter += 1

print('Predicting data-------')
pred = model.predict(x_val)
df_pred = pd.DataFrame(argmax(pred, 1))
df_pred['real'] = argmax(y_val, 1)
print('Prediction on Val errors: ' + str(sum(df_pred[0] != df_pred['real'])))


# Test data
def get_label_name(index):
    return lb_style.classes_[index]


# Test
# Encode Train data
encoded_seqs_test = t.texts_to_sequences(test_data['text_cleaned'])

# Pad Train Data
encoded_seqs_test_pad = pad_sequences(encoded_seqs_test, maxlen=max_len_encoded, dtype='int32', padding='pre', truncating='pre', value=0.0)

pred = model.predict(encoded_seqs_test_pad)
df_pred = pd.DataFrame(argmax(pred, 1))
out_data = pd.DataFrame(df_pred[0].map(get_label_name))
out_data.columns = ['category']
out_data.index.name = 'id'

out_data.to_csv("./results.csv", header=True)

# Save model
print('Saving the Model')
model_json = model.to_json()
file_model_name = model_name
with open('./' + file_model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./' + file_model_name + '.h5')

