
import re
import string

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import *
from keras.preprocessing.text import Tokenizer
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


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
data = pd.read_csv(input_path, nrows=700)

data['title_cleaned'] = data['title'].apply(custom_text_format)


docs = data['title_cleaned']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)


# summarize what was learned
print(t.word_counts)
print(t.document_count)
# print(t.word_index)
# print(t.word_docs)


encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)

#
# # categories - dummy
# cats = data['category']
# pd.get_dummies(data, columns=['category']).head()
#
# # categories - sklearn
# lb_make = LabelEncoder()
# data["category_code"] = lb_make.fit_transform(data["category"])


# categories - sklearn
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(data["category"])


print('X Data Size')
print(encoded_docs.shape)
print('Y Data Size')
print(lb_results.shape)



Y_pre = lb_results
X_pre = encoded_docs


x_train, x_val, y_train, y_val = train_test_split(X_pre, Y_pre, test_size=0.3)


model = Sequential()
model.add(Dense(2048, activation='relu', input_dim=x_train.shape[1]))
# model.add(Dense(2048, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# model.fit(x_train, y_train,
#                      validation_data=[x_val, y_val],
#                      epochs=1,
#                      batch_size=16,
#                      verbose=2)


score_all = model.evaluate(X_pre, Y_pre, verbose=0)
score_all_test = model.evaluate(x_val, y_val, verbose=0)
print('All data acc: ' + str(score_all[1]))
print('Val data acc: ' + str(score_all_test[1]))

while score_all[1] <= 0.999:
    # model.fit(X, Y, epochs=epochs, batch_size=5, verbose=2, callbacks=[tbCallBack])
    # model.fit(X, Y, epochs=epochs, batch_size=32, verbose=2)
    model.fit(x_train, y_train,
                         validation_data=[x_val, y_val],
                         epochs=1,
                         batch_size=32,
                         verbose=2)
    score_all = model.evaluate(X_pre, Y_pre, verbose=2)
    score_all_test = model.evaluate(x_val, y_val, verbose=0)
    print('All data acc: ' + str(score_all[1]))
    print('Test data acc: ' + str(score_all_test[1]))
    print('--------------')



# prediction_all = np.round(model.predict(X_pre))




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