
# Summary data needed

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, CSVLogger

import os
import time
import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1337)  # for reproducibility


file_name = "summary_spanish_cnn_256_128_64_lstm_250_b_19results.csv"


base_path = "./test_models_results"

# Analysis using n_batch
model_name_base = "portuguese_cnn_64_128_lstm_200_b_"
n_batch = 19


# Loading summary data
summary_info = pd.read_csv(base_path + "/" + file_name)

# worst train data set
# n_subset = 100

condition = summary_info['state_prediction'] < 0.5
subset = summary_info[condition]

print(subset)
print('Count subset: ' + str(subset['count'].sum()))

reinforce_subset_cats = subset['class_real']


print("Reading pkl SPANISH")
start = time.time()
train_data = pd.read_pickle('./data/train_subset_spanish.pkl')

condition_all_data = train_data['category'] == reinforce_subset_cats[0]
for k in range(1,len(reinforce_subset_cats)):
    # print(k)
    condition_all_data |= (train_data['category'] == reinforce_subset_cats[k])


# condition_all_data
train_data_subset = train_data[condition_all_data]
print("Data subset size: " + str(len(train_data_subset)))


print("Summary SPANISH Train size: ")
print(train_data_subset.shape)


# Loading Tokenizer and Binarizer
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
json_file = open('./models/portuguese/' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/portuguese/' + model_name + ".h5")
print("Loaded model from disk")
print(model.summary())


# Creating train data set
encoded_seqs = t.texts_to_sequences(train_data_subset['text_cleaned'])
lb_results = lb_style.transform(train_data_subset["category"])

# lb_results = lb_style.fit_transform(train_data_subset["category"])

max_len_encoded = model.input_shape[1]
encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=max_len_encoded,
                                 dtype='int32', padding='pre', truncating='pre', value=0.0)


Y_pre = lb_results
X_pre = encoded_seqs_pad

# Saving train_data and labels: X and Y
print("Saving train data")
batches_path = './train_data/batches/spanish'
n_batch = 20

np.save(batches_path + '/' + 'x_train_' + str(n_batch) + '.npy', X_pre)
np.save(batches_path + '/' + 'y_train_' + str(n_batch) + '.npy', Y_pre)

