
import time
import numpy as np
import pandas as pd

from libs.cleaning import custom_text_format

np.random.seed(1337)  # for reproducibility


# input_path_test = "/Users/Juan/Downloads/mercadolibre_data/test.csv"

input_path_test = "../test.csv"

print("Loading data")
test_data = pd.read_csv(input_path_test)

# Clear Data Test
print("Cleaning data")
start = time.time()
test_data['text_cleaned'] = test_data['title'].apply(custom_text_format)
# test_data_subset = pd.DataFrame(test_data['text_cleaned'])
stop = time.time()
print(stop - start)


lang_groups = test_data.groupby('language')
print(lang_groups.count())


# Save as pkl
print('Saving Spanish pkl')
# Save as pkl
start = time.time()
lang_groups.get_group('spanish').to_pickle('./data/test_subset_spanish.pkl')
stop = time.time()
print(stop - start)


# Save as h5
print('Saving Portuguese pkl')
# Save as pkl
start = time.time()
lang_groups.get_group('portuguese').to_pickle('./data/test_subset_portuguese.pkl')
stop = time.time()
print(stop - start)

