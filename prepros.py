
import re
import pandas as pd
import numpy as np
import keras


def clean_review(text_arg):
    # Strip HTML tags
    # text = re.sub('<[^<]+?>', ' ', text)
    text_arg = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_arg)
    return text_arg


def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))


def custom_text_format(text_arg):
    words = text_arg.split()
    text_fixed = ''

    for w in words:
        # print(w)
        # print(has_numbers(w))
        if has_numbers(w):
            try:
                number = float(w)
                # print('it has only numbers')
                text_fixed += 'a_number'
            except:
                text_fixed += 'a_mixed_number'
                # print('it has mixed words')
        else:
            text_fixed += w

        text_fixed += ' '

    text_fixed = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_fixed)

    return text_fixed.lower()




input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
data = pd.read_csv(input_path, nrows=100)

data['title_cleaned'] = data['title'].apply(custom_text_format)

#
#
# text = data['title'][0]
#
# data['title_cleaned'][80]