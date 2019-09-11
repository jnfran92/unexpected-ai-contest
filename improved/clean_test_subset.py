
import time
import numpy as np
import pandas as pd

from libs.cleaning import custom_text_format

np.random.seed(1337)  # for reproducibility


input_path_test = "/Users/Juan/Downloads/mercadolibre_data/test.csv"

# input_path_test = "../../test.csv"

print("Loading data")
test_data = pd.read_csv(input_path_test)

# Clear Data Test
print("Cleaning data")
start = time.time()
test_data['text_cleaned'] = test_data['title'].apply(custom_text_format)
test_data_subset = pd.DataFrame(test_data['text_cleaned'])
stop = time.time()
print(stop - start)

# Save to pkl
print('Saving pkl')
test_data_subset.to_pickle('./data/test_subset.pkl')

# Save to h5
print('Saving h5')
test_data_subset.to_hdf('./data/test_subset.h5', 'test_data')
