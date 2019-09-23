
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

# Analysis using n_batch
model_name_base = "spanish_cnn_256_128_64_lstm_250_b_"
n_batch = 19

# Read Test
# test_data = pd.read_pickle('./data/test_subset_spanish.pkl').reset_index(drop=True)
test_data = pd.read_pickle('./data/test_subset_spanish.pkl')

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
print("Loading Model")
model_name = model_name_base + str(n_batch)
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
print("Creating results using test data")
pred = model.predict(encoded_seqs_pad)

df_pred = pd.DataFrame(argmax(pred, 1))
out_data = pd.DataFrame(df_pred[0].map(get_label_name))
out_data.columns = ['category']
# out_data.index.name = 'id'
out_data['id'] = test_data['id']


out_data['title'] = test_data['text_cleaned']

out_data.to_csv("./test_models_results/test_" + model_name + "results.csv", header=True)

# Summary data and errors
print("Summarizing data using Validation Data")
print("Reading validation SPANISH")
batches_path = './train_data/batches/spanish'
x_val0 = np.load(batches_path + '/' + 'x_val' + str(0) + '.npy')
x_val1 = np.load(batches_path + '/' + 'x_val' + str(1) + '.npy')

y_val0 = np.load(batches_path + '/' + 'y_val' + str(0) + '.npy')
y_val1 = np.load(batches_path + '/' + 'y_val' + str(1) + '.npy')

x_val = np.concatenate((x_val0, x_val1))
y_val = np.concatenate((y_val0, y_val1))

print('Predicting data-------')
pred_val = model.predict(x_val)

df_pred_val = pd.DataFrame(argmax(pred_val, 1))
df_pred_val['real'] = argmax(y_val, 1)
df_pred_val['state_prediction'] = df_pred_val[0] == df_pred_val['real']
df_pred_val['text'] = t.sequences_to_texts(x_val)
df_pred_val['real_class'] = df_pred_val['real'].map(get_label_name)
df_pred_val['pred_class'] = df_pred_val[0].map(get_label_name)

df_pred_val = df_pred_val.sort_values(by= 'real')
df_pred_val.to_csv("./test_models_results/val_" + model_name + "results.csv", header=True)


grs = df_pred_val.groupby('real')
grs_summary = grs.mean()
grs_summary['count'] = grs.count()[0]
grs_summary['class_real'] = grs_summary.index.map(get_label_name)
grs_summary = grs_summary.sort_values(by="state_prediction")


grs_summary.to_csv("./test_models_results/summary_" + model_name + "results.csv", header=True)
