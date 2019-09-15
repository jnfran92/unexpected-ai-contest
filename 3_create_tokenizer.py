
import os
import time


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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

print("Reading pkl")
start = time.time()
train_data = pd.read_pickle('./data/train_subset_spanish.pkl')
test_data = pd.read_pickle('./data/test_subset_spanish.pkl')
stop = time.time()
print(stop - start)


# print("Reading h5")
# start = time.time()
# train_data = pd.read_hdf('./data/train_subset.h5', 'train_data')
# test_data = pd.read_hdf('./data/test_subset.h5', 'test_data')
# stop = time.time()
# print(stop - start)


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
