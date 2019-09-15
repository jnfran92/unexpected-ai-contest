
import os
import time
import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

print("Reading pkl PORTUGUESE")
start = time.time()
train_data = pd.read_pickle('./data/train_subset_portuguese.pkl')
test_data = pd.read_pickle('./data/test_subset_portuguese.pkl')
stop = time.time()
print(stop - start)


print("Summary SPANISH Train and test size: ")
print(train_data.shape)
print(test_data.shape)

print("Summary of groups PORTUGUESE")
print(train_data.groupby('category').count())

docs_global = pd.concat([train_data['text_cleaned'], test_data['text_cleaned']]).reset_index(drop=True)
print('Size of train + test: ' + str(docs_global.shape))

# create the tokenizer
print('Creating the tokenizer PORTUGUESE')
t = Tokenizer(char_level=True)
t.fit_on_texts(docs_global)

print("Summary of vocabulary at char level: ")
print(t.word_counts)

print('Saving the Tokenizer PORTUGUESE')
# saving
with open('./tokenizer/tokenizer_portuguese.pickle', 'wb') as handle:
    pickle.dump(t, handle, protocol=pickle.HIGHEST_PROTOCOL)

