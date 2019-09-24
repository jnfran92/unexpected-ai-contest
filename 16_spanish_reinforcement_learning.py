
# Summary data needed

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, CSVLogger


file_name = "summary_spanish_cnn_256_128_64_lstm_250_b_19results.csv"


base_path = "./test_models_results"

summary_info = pd.read_csv(base_path + "/" + file_name)


