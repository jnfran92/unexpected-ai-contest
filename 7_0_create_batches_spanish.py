
import os
import time

import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1337)  # for reproducibility

print("Reading npy SPANISH")
n_part = 0
start = time.time()
X_pre = np.load('./train_data/X' + str(n_part) + '_spanish.npy')
Y_pre = np.load('./train_data/Y' + str(n_part) + '_spanish.npy')
stop = time.time()
print(stop - start)


print("X Train data size")
print(X_pre.shape)
print("Y Train data size")
print(Y_pre.shape)

# Split Data
x_train, x_val, y_train, y_val = train_test_split(X_pre, Y_pre, test_size=0.3)


# Saving validation data
print("Saving validation data")
batches_path = './train_data/batches/spanish'
np.save(batches_path + '/' + 'x_val' + str(n_part) + '.npy', x_val)
np.save(batches_path + '/' + 'y_val' + str(n_part) + '.npy', y_val)

# Creating batches
n_batches = 10

x_train_batches = np.array_split(x_train, n_batches)
y_train_batches = np.array_split(y_train, n_batches)

for x in range(n_batches):
    print("Saving " + str(x) + " batch")
    print('x_train batch size: ' + str(x_train_batches[x].shape))
    print('y_train batch size: ' + str(y_train_batches[x].shape))
    np.save(batches_path + '/' + 'x_train_' + str(x) + '.npy', x_train_batches[x])
    np.save(batches_path + '/' + 'y_train_' + str(x) + '.npy', y_train_batches[x])




