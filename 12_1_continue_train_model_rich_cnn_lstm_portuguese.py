import os
import sys
import time

import numpy as np
from keras.callbacks import EarlyStopping, CSVLogger
from keras.models import model_from_json
from keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(1337)  # for reproducibility

batches_path = './train_data/batches/portuguese'
# n_batch = 1  # Batch to train

if (len(sys.argv)) < 5:
    print('ERROR: Args error, 4 args needed')
    sys.exit()


# nohup python3 12_1_continue_train_model_rich_cnn_lstm_portuguese.py 0 5 6 0.0001 > last_log_portuguese.out &

n_batch = int(sys.argv[1])
print("n specific batch to train SPANISH:  " + str(n_batch))

n_model = int(sys.argv[2])
print("n model to re-force training SPANISH:  " + str(n_model))

out_model = int(sys.argv[3])
print("Out model label trained SPANISH:  " + str(out_model))

custom_lr = float(sys.argv[4])
print("Custom Lr SPANISH:  " + str(custom_lr))

#
#
# n_batch = int(sys.argv[1])
# print("n_batch to train PORTUGUESE:  " + str(n_batch))
#
#
# custom_lr = float(sys.argv[2])
# print("Custom Lr PORTUGUESE:  " + str(custom_lr))

model_name_base = "portuguese_cnn_dense_2_b_"

file_model_name = model_name_base + str(out_model)

print("Reading train batch and val portuguese")
print("Reading validation")
x_val0 = np.load(batches_path + '/' + 'x_val' + str(0) + '.npy')
# x_val1 = np.load(batches_path + '/' + 'x_val' + str(1) + '.npy')

y_val0 = np.load(batches_path + '/' + 'y_val' + str(0) + '.npy')
# y_val1 = np.load(batches_path + '/' + 'y_val' + str(1) + '.npy')

# x_val = np.concatenate((x_val0, x_val1))
# y_val = np.concatenate((y_val0, y_val1))

x_val = x_val0
y_val = y_val0

print("Reading batch")
start = time.time()
x_train = np.load(batches_path + '/' + 'x_train_' + str(n_batch) + '.npy')
y_train = np.load(batches_path + '/' + 'y_train_' + str(n_batch) + '.npy')
stop = time.time()
print(stop - start)


# print("Loading Model")
model_name = model_name_base + str(n_model)
print("Loading model: " + model_name)
# load json and create model
json_file = open('./models/portuguese/' + model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('./models/portuguese/' + model_name + ".h5")
print("Loaded model from disk")

# Common operation
# adam_opt = Adam(learning_rate=custom_lr)
adam_opt = Adam(lr=custom_lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Create Callback
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=6,
                           verbose=1)

csv_logger = CSVLogger(filename="./logs/" + file_model_name + ".csv")

# Train
fit_data = model.fit(x_train, y_train,
                     validation_data=[x_val, y_val],
                     epochs=15,
                     batch_size=128,
                     verbose=2,
                     callbacks=[early_stop, csv_logger])

#
# print('Predicting data-------')
# pred = model.predict(x_val)
# df_pred = pd.DataFrame(argmax(pred, 1))
# df_pred['real'] = argmax(y_val, 1)
# print('Prediction on Val errors: ' + str(sum(df_pred[0] != df_pred['real'])))
#

# Save model
print('Saving the Model')
model_json = model.to_json()

with open('./models/portuguese/' + file_model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('./models/portuguese/' + file_model_name + '.h5')

