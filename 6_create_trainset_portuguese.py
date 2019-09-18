
import os
import pickle
import time

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

print("Reading pkl PORTUGUESE")
start = time.time()
train_data = pd.read_pickle('./data/train_subset_portuguese.pkl')
test_data = pd.read_pickle('./data/test_subset_portuguese.pkl')
stop = time.time()
print(stop - start)

t = Tokenizer()
print("Loading the tokenizer")
with open('./tokenizer/tokenizer_portuguese.pickle', 'rb') as handle:
    t = pickle.load(handle)

print("Summary portuguese Train and test size: ")
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


# Saving Binarizer:
print('Saving binarizer PORTUGUESE')
with open('./binarizer/binarizer_portuguese.pickle', 'wb') as handle:
    pickle.dump(lb_style, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('X Data Size')
print(encoded_seqs_pad_scaled.shape)
print('Y Data Size')
print(lb_results.shape)


Y_pre = lb_results
X_pre = encoded_seqs_pad_scaled
# Saving train_data and labels: X and Y
X_pre_size_half = int(X_pre.shape[0]/2)
print("Saving train data 1")
np.save("./train_data/X0_portuguese.npy", X_pre[0:X_pre_size_half])
np.save("./train_data/Y0_portuguese.npy", Y_pre[0:X_pre_size_half])

print("Saving train data 2")
np.save("./train_data/X1_portuguese.npy", X_pre[X_pre_size_half:])
np.save("./train_data/Y1_portuguese.npy", Y_pre[X_pre_size_half:])
