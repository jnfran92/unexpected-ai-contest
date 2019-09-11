
import os
import re
import string
import pandas as pd
import numpy as np
from multiprocessing import Pool

np.random.seed(1337)  # for reproducibility


def has_numbers(input_string):
    """ test if the string is a number"""
    return bool(re.search(r'\d', input_string))


def custom_text_format(text_arg):
    """Apply a custom cleaning process: remove special characters,
    and set numbers and numbers-words with tags"""
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
                    text_fixed += 'NUM'
                    # text_fixed += ''
                except:
                    text_fixed += 'MIX'
                    # print('it has mixed words')
                    # text_fixed += ''
            else:
                text_fixed += w


        else:
            text_fixed += ''

        text_fixed += ' '
    # text_fixed = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_fixed)

    return text_fixed.lower()


data_path = "/Users/Juan/Downloads/mercadolibre_data/train_chunks"
input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
input_path_test = "/Users/Juan/Downloads/mercadolibre_data/test.csv"

# data_path = "../../"
# input_path = "../../train.csv"
# input_path_test = "../../test.csv"


def custom_read_csv(filename):
    """converts a filename to a pandas dataframe"""
    # print(filename)
    t = data_path + '/' + filename
    return pd.read_csv(t, header=None)


# start a pool for parallel process
pool = []
if __name__ == '__main__':
    pool = Pool()  # or whatever your hardware can support


files = os.listdir(data_path)
file_list = [filename for filename in files if filename.split('.')[1] == 'csv']


df_list = pool.map(custom_read_csv, file_list)

train_data = pd.concat(df_list,  ignore_index=True, axis=0)
train_data.columns = ['title', 'confidence', 'lang', 'category']
# n_rows = 20e6
# train_data = pd.read_csv(input_path, nrows=n_rows)

test_data = pd.read_csv(input_path_test)

# Subset
gs = train_data.groupby('category')

gs_summary = gs.count().sort_values(by='title')
print(gs_summary)

# Get subset to test n groups with large number of samples by group
n_groups = 10
gs_cats = gs_summary.iloc[:, :].index
print('Groups selected: ')
print(gs_cats)

c_temp = (train_data['category'] == '')
for c in gs_cats:
    # print(c)
    c_temp |= (train_data['category'] == c)

c_all = c_temp
train_data_subset = train_data[c_all].reset_index(drop=True)
print('Train_data subset Shape: ' + str(train_data_subset.shape))

# Clear Data Train
train_data_subset['text_cleaned'] = train_data_subset['title'].apply(custom_text_format)
train_data_subset.to_pickle("./train_subset")

# Clear Data Test
test_data['text_cleaned'] = test_data['title'].apply(custom_text_format)
test_data_subset = pd.DataFrame(test_data['text_cleaned'])
test_data_subset.to_pickle('./test_subset')
