
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import time
# import swifter

from libs.cleaning import custom_text_format

np.random.seed(1337)  # for reproducibility


# data_path = "/Users/Juan/Downloads/mercadolibre_data/train_chunks"

data_path = "../train_chunks"


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

n_limit = 0          # getting all the data with zero
file_list_limited = file_list[n_limit:]

print("Reading data")
start = time.time()
df_list = pool.map(custom_read_csv, file_list_limited)
stop = time.time()
print(stop - start)

train_data = pd.concat(df_list,  ignore_index=True, axis=0)
train_data.columns = ['title', 'confidence', 'lang', 'category']

# Get all groups
train_data_subset = train_data
print('Train_data subset Shape: ' + str(train_data_subset.shape))

# Clear Data Train
print('Cleaning train data')
start = time.time()
train_data_subset['text_cleaned'] = train_data_subset['title'].apply(custom_text_format)
stop = time.time()
print(stop - start)

# Group by language
lang_groups = train_data_subset.groupby('lang')
print(lang_groups.count())


# Save as pkl
print('Saving Spanish pkl')
# Save as pkl
start = time.time()
lang_groups.get_group('spanish').to_pickle('./data/train_subset_spanish.pkl')
stop = time.time()
print(stop - start)


# Save as h5
print('Saving Portuguese pkl')
# Save as pkl
start = time.time()
lang_groups.get_group('portuguese').to_pickle('./data/train_subset_portuguese.pkl')
stop = time.time()
print(stop - start)
