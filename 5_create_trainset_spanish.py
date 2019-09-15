
import os
import time
import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

print("Reading pkl SPANISH")
start = time.time()
train_data = pd.read_pickle('./data/train_subset_spanish.pkl')
test_data = pd.read_pickle('./data/test_subset_spanish.pkl')
stop = time.time()
print(stop - start)

t = Tokenizer()
print("Loading the tokenizer")
with open('./tokenizer/tokenizer_spanish.pickle', 'rb') as handle:
    t = pickle.load(handle)

print("Summary SPANISH Train and test size: ")
print(train_data.shape)
print(test_data.shape)


# Creating train data set
encoded_seqs = t.texts_to_sequences(train_data['text_cleaned'])
encoded_seqs_dummy = t.texts_to_sequences(test_data['text_cleaned'])

s1 = np.max([len(item) for item in encoded_seqs])
s2 = np.max([len(item) for item in encoded_seqs_dummy])
s3 = np.max([s1, s2])

# Pad Train Data
max_len_encoded = s3
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


Y_pre = lb_results
X_pre = encoded_seqs_pad_scaled
# Saving train_data and labels: X and Y
print("Saving train data")
np.save("./train_data/X_spanish.npy", X_pre)
np.save("./train_data/Y_spanish.npy", Y_pre)
