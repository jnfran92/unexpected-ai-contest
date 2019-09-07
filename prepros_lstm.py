
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
                    text_fixed += 'anumber'
                    # text_fixed += ''
                except:
                    text_fixed += 'amixednumber'
                    # print('it has mixed words')
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

train_data = pd.read_csv(input_path, nrows=1500)

test_data = pd.read_csv(input_path_test, nrows=1500)

train_data_text_cleared = train_data['title'].apply(custom_text_format)

test_data_text_cleared = test_data['title'].apply(custom_text_format)

docs_global = pd.concat([train_data_text_cleared, test_data_text_cleared])
docs_global = docs_global.reset_index(drop=True)


# create the tokenizer
t = Tokenizer()
t.fit_on_texts(docs_global)

# summarize what was learned
print('Test plus Train data documents:')
print(t.document_count)

# Encode Train data
encoded_seqs = t.texts_to_sequences(train_data_text_cleared)
print(encoded_seqs)

# Pad Train Data
encoded_seqs_pad = pad_sequences(encoded_seqs, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)

# Scaler
scaler = StandardScaler().fit(encoded_seqs_pad)
encoded_seqs_pad_scaled = scaler.transform(encoded_seqs_pad)
# encoded_seqs_pad_scaled = encoded_seqs_pad

print("Encoded Seqs Shape")
print(encoded_seqs_pad.shape)

# categories - sklearn
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(train_data["category"])


print('X Data Size')
print(encoded_seqs_pad_scaled.shape)
print('Y Data Size')
print(lb_results.shape)

shape_encoded = encoded_seqs_pad_scaled.shape

# all data (train)
Y_pre = lb_results
X_pre = encoded_seqs_pad_scaled.reshape([shape_encoded[0], shape_encoded[1], 1])

# Split Data
x_train, x_val, y_train, y_val = train_test_split(X_pre, Y_pre, test_size=0.3)

# Create model
model = Sequential()
model.add(
    Conv1D(filters=128, kernel_size=12, padding='causal', activation='relu',
           input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))

# model.add(
#     Conv1D(filters=8, kernel_size=4, padding='causal', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128))
model.add(Dense(y_train.shape[1], activation='softmax'))


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())

score_all = model.evaluate(X_pre, Y_pre, verbose=0)
score_all_test = model.evaluate(x_val, y_val, verbose=0)
# print('All data acc: ' + str(score_all[1]))
# print('Val data acc: ' + str(score_all_test[1]))
print('All data acc: ' + str(score_all[1]))
print('Val data acc: ' + str(score_all_test[1]))

counter = 1
while score_all[1] <= 0.999:
    print('counter steps: ' + str(counter))
    # model.fit(X, Y, epochs=epochs, batch_size=5, verbose=2, callbacks=[tbCallBack])
    # model.fit(X, Y, epochs=epochs, batch_size=32, verbose=2)
    model.fit(x_train, y_train,
                         validation_data=[x_val, y_val],
                         epochs=1,
                         batch_size=8,
                         verbose=2)

    score_all = model.evaluate(X_pre, Y_pre, verbose=2)
    score_all_test = model.evaluate(x_val, y_val, verbose=0)
    print('All data acc: ' + str(score_all[1]))
    print('Val data acc: ' + str(score_all_test[1]))
    print('--------------')
    counter += 1


# Test data
def get_label_name(index):
    return lb_style.classes_[index]


pred = model.predict(X_pre)
df_pred = pd.DataFrame(argmax(pred, 1))
df_pred['real'] = argmax(Y_pre, 1)

condition = ((df_pred[0] - df_pred['real']) == 0)

train_data['prediction'] = df_pred[0].map(get_label_name)
train_data['prediction_state'] = condition
train_data['title_cleared'] = train_data_text_cleared



# Prediction on Test Data
encoded_docs_test = t.texts_to_sequences(test_data_text_cleared)
encoded_docs_test_pad = pad_sequences(encoded_docs_test, maxlen=encoded_seqs_pad.shape[1], dtype='int32', padding='pre', truncating='pre', value=0.0)
shp_temp = encoded_docs_test_pad.shape
encoded_docs_test_pad = scaler.transform(encoded_docs_test_pad).reshape([shp_temp[0], shp_temp[1], 1])

pred = model.predict(encoded_docs_test_pad)
df_pred_val = pd.DataFrame(argmax(pred, 1))
test_data['prediction'] = df_pred_val.apply(get_label_name)




# out = model.fit(x_train, y_train,
#                 validation_data=[x_val, y_val],
#                 epochs=1,
#                 batch_size=32,
#                 verbose=2)


# model.fit(x_train, y_train,
#                      validation_data=[x_val, y_val],
#                      epochs=1,
#                      batch_size=16,
#                      verbose=2)


#
#
#
# encoded_docs_test = t.texts_to_matrix(test_data_text_cleared, mode='count')
# print(encoded_docs)
#
#
# pred = model.predict(x_train)
# # print(argmax(pred, 1))
# # print(argmax(y_train, 1))
# test_data_text_cleared['real_id'] = argmax(y_train, 1)
# test_data_text_cleared['pred_id'] = argmax(pred, 1)
#
#
#
# df_pred = pd.DataFrame(argmax(pred, 1))
# df_pred['real'] = argmax(y_train, 1)
#
#
# # prediction_all = np.round(model.predict(X_pre))




#
# score_all = model.evaluate(X_pre, Y_pre, verbose=0)
# print('Score All')
# print(score_all[1])
#
#
#
#
#
#
# print('All Confution matrix:')
# prediction_all = np.round(model.predict(X_pre))
# print(confusion_matrix(Y_pre, prediction_all))
#
#
#
#
# pred = model.predict(x_train)

# score = model.evaluate(x_test, y_test, batch_size=128)

#
# print('ROC AUC Score All: ' + str(roc_auc_score(y_val, model.predict(x_val))))
#
#
# print('All data Confution matrix:')
# prediction_all = np.round(model.predict(X_pre))
# cm = confusion_matrix(argmax(Y_pre, 1),
#                        argmax(prediction_all, 1))
# print(cm)
#
# print('Test Confution matrix:')
# prediction_all2 = np.round(model.predict(x_val))
# cm_val = confusion_matrix(argmax(y_val, 1),
#                        argmax(prediction_all2, 1))




#
#
#
# X_train, X_test, y_train, y_test = train_test_split(data['title_cleaned'], data['category'], test_size=0.2)
#
#
#
#
#
#
# vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
#                              lowercase=True, min_df=3, max_df=0.9, max_features=5000)
#
#
# X_train_onehot = vectorizer.fit_transform(X_train)
#
#
#
# # define the document
# text = 'The quick brown fox jumped over the lazy dog.'
# # tokenize the document
# result = text_to_word_sequence(text)
# print(result)
#
#
#
#
#
# #
# # text = data['title'][0]
# #
# # result = text.translate(str.maketrans('', '', string.punctuation))
#
# #
# #
# # text = data['title'][0]
# #
# # data['title_cleaned'][80]