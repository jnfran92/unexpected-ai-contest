
import re
import pandas as pd
import numpy as np
import keras
import string

def clean_review(text_arg):
    # Strip HTML tags
    # text = re.sub('<[^<]+?>', ' ', text)
    text_arg = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_arg)
    return text_arg


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


text = data['title'][0]

result = text.translate(str.maketrans('', '', string.punctuation))

#
#
# text = data['title'][0]
#
# data['title_cleaned'][80]