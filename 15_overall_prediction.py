
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Read data
folder_path = './test_models_results'

spanish_name = 'test_spanish_cnn_256_128_64_lstm_250_b_19results'
portuguese_name = 'test_portuguese_cnn_64_128_lstm_200_b_17results'

d_spanish = pd.read_csv(folder_path + '/' + spanish_name + '.csv')
d_portuguese = pd.read_csv(folder_path + '/' + portuguese_name + '.csv')


final_prediction = pd.concat((d_spanish[['id', 'category']], d_portuguese[['id', 'category']]))

final_prediction = final_prediction.sort_values(by='id')

final_prediction.to_csv("./test_models_results/final_" + spanish_name + "_plus_" + portuguese_name + '.csv',
                        header=True, index=False)
