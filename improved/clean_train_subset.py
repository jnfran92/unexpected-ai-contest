
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import time

from libs.cleaning import custom_text_format

np.random.seed(1337)  # for reproducibility


data_path = "/Users/Juan/Downloads/mercadolibre_data/train_chunks"
input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
# input_path_test = "/Users/Juan/Downloads/mercadolibre_data/test.csv"

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

file_list_limited = file_list[40:]

df_list = pool.map(custom_read_csv, file_list_limited)

train_data = pd.concat(df_list,  ignore_index=True, axis=0)
train_data.columns = ['title', 'confidence', 'lang', 'category']

# test_data = pd.read_csv(input_path_test)

# Subset
# gs = train_data.groupby('category')
#
# gs_summary = gs.count().sort_values(by='title')
# print(gs_summary)

# Get subset to test n groups with large number of samples by group
# n_groups = 10
# gs_cats = gs_summary.iloc[:, :].index
# print('Groups selected: ')
# print(gs_cats)

# Get all groups
train_data_subset = train_data
print('Train_data subset Shape: ' + str(train_data_subset.shape))

# Clear Data Train
print('Cleaning train data')
start = time.time()
train_data_subset['text_cleaned'] = train_data_subset['title'].apply(custom_text_format)
stop = time.time()
print(stop - start)


print('Saving pkl')
# Save as pkl
start = time.time()
train_data_subset.to_pickle("./data/train_subset.pkl")
stop = time.time()
print(stop - start)

# Save as h5
print('Saving h5')
# Save as pkl
start = time.time()
train_data_subset.to_hdf('./data/train_subset.h5', 'train_data')
stop = time.time()
print(stop - start)

#
# # Clear Data Test
# test_data['text_cleaned'] = test_data['title'].apply(custom_text_format)
# test_data_subset = pd.DataFrame(test_data['text_cleaned'])
# test_data_subset.to_pickle('./test_subset')
