
import re
import string

import numpy as np

np.random.seed(1337) # for reproducibility
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences


def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))


def custom_text_format(text_arg):
    text_arg = text_arg.translate(str.maketrans('', '', string.punctuation))
    words = text_arg.split()
    text_fixed = ''

    for w in words:
        # print(w)
        # print(has_numbers(w))
        if len(w) > 2:
            if has_numbers(w):
                try:
                    number = float(w)
                    # print('it has only numbers')
                    # text_fixed += 'anumber'
                    text_fixed += ''
                except:
                    # text_fixed += 'amixednumber'
                    # print('it has mixed words')
                    text_fixed += ''
            else:
                text_fixed += w


        else:
            text_fixed += ''

        text_fixed += ' '
    # text_fixed = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_fixed)

    return text_fixed.lower()


input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
input_path_test = "/Users/Juan/Downloads/mercadolibre_data/test.csv"
# input_path = "./train.csv"

train_data = pd.read_csv(input_path, nrows=15000)

test_data = pd.read_csv(input_path_test, nrows=1500)

train_data_text_cleared = train_data['title'].apply(custom_text_format)

test_data_text_cleared = test_data['title'].apply(custom_text_format)

docs_global = pd.concat([train_data_text_cleared, test_data_text_cleared])
docs_global = docs_global.reset_index(drop=True)

# Subset
gs = train_data.groupby('category')

gs_summary = gs.count()
print(gs_summary.sort_values(by='title'))



