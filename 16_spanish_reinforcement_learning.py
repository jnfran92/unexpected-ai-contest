
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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1337)  # for reproducibility


file_name = "summary_spanish_cnn_256_128_64_lstm_250_b_19results.csv"


base_path = "./test_models_results"

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


train_data['category']
