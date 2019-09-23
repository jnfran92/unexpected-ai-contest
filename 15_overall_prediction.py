
import os
import time
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

from libs.cleaning import custom_text_format

np.random.seed(1337)  # for reproducibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Read Test Spanish

#
# test_data_spanish = pd.read_pickle('./data/test_subset_spanish.pkl')
#
# print("Reading Tokenizer SPANISH")
# t_spanish = Tokenizer()
# print("Loading the tokenizer")
# with open('./tokenizer/tokenizer_spanish.pickle', 'rb') as handle:
#     t_spanish = pickle.load(handle)
#
#
# print("Reading Binarizer SPANISH")
# lb_style_spanish = LabelBinarizer()
# with open('./binarizer/binarizer_spanish.pickle', 'rb') as handle:
#     lb_style_spanish = pickle.load(handle)
#
#
#
# # Loading Model
# print("Loading Model")
# model_name = model_name_base + str(n_batch)
# print("Loading model: " + model_name)
# # load json and create model
# json_file = open('./models/spanish/' + model_name + '.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# # load weights into new model
# model.load_weights('./models/spanish/' + model_name + ".h5")
# print("Loaded model from disk")
# print(model.summary())
#
# # Creating test data set
# print("Creating test data set")
# encoded_seqs = t.texts_to_sequences(test_data['text_cleaned'])
#
# # Pad Test Data
# print("Pad Test Data")
# max_len_encoded = model.input_shape[1]
# encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=max_len_encoded, dtype='int32', padding='pre', truncating='pre', value=0.0)
#
#
#
#
#
#
# # Read Test Portuguese
# test_data_portuguese = pd.read_pickle('./data/test_subset_portuguese.pkl')
#
# print("Reading Tokenizer PORTUGUESE")
# t_portuguese = Tokenizer()
# print("Loading the tokenizer")
# with open('./tokenizer/tokenizer_portuguese.pickle', 'rb') as handle:
#     t_portuguese = pickle.load(handle)
#
#
# print("Reading Binarizer PORTUGUESE")
# lb_style_portuguese = LabelBinarizer()
# with open('./binarizer/binarizer_portuguese.pickle', 'rb') as handle:
#     lb_style_portuguese = pickle.load(handle)
#

