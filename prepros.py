
import re
import pandas as pd
import numpy as np
import keras
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import *
from keras.losses import *
from keras.activations import *

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
data = pd.read_csv(input_path, nrows=300)

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

# categories - sklearn
lb_make = LabelEncoder()
data["category_code"] = lb_make.fit_transform(data["category"])


# categories - sklearn
lb_style = LabelBinarizer()
lb_results = lb_style.fit_transform(data["category"])


print(encoded_docs.shape)
print(lb_results.shape)



Y_pre = lb_results
X_pre = encoded_docs


x_train, x_val, y_train, y_val = train_test_split(X_pre, Y_pre, test_size=0.3)


model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='sigmoid'))

model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train,
                     validation_data=[x_val, y_val],
                     epochs=32,
                     batch_size=32,
                     verbose=2)



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