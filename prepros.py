
import re
import pandas as pd
import numpy as np
import keras
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from keras.preprocessing.text import text_to_word_sequence


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
                    text_fixed += 'a_number'
                    # text_fixed += ''
                except:
                    text_fixed += 'a_mixed_number'
                    # print('it has mixed words')
            else:
                text_fixed += w


        else:
            text_fixed += ''

        text_fixed += ' '
    # text_fixed = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_fixed)

    return text_fixed.lower()


input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
data = pd.read_csv(input_path, nrows=100)

data['title_cleaned'] = data['title'].apply(custom_text_format)

X_train, X_test, y_train, y_test = train_test_split(data['title_cleaned'], data['category'], test_size=0.2)


vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)


X_train_onehot = vectorizer.fit_transform(X_train)



# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)



from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)



#
# text = data['title'][0]
#
# result = text.translate(str.maketrans('', '', string.punctuation))

#
#
# text = data['title'][0]
#
# data['title_cleaned'][80]