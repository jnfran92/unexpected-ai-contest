import re
import pandas as pd
import numpy as np
import keras


def clean_review(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)

    # Remove special characters
    text = re.sub('[^a-zA-Z0-9 \n\.]', '', text)

    # Replace numbers
    text = re.sub(r'[0-9]+', 'a_number', text)

    # Strip escaped quotes
    text = text.replace('\\"', '')

    # Strip quotes
    text = text.replace('"', '')

    return text


def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))


input_path = "/Users/Juan/Downloads/mercadolibre_data/train.csv"
data = pd.read_csv(input_path, nrows=100)

data['title_cleaned'] = data['title'].apply(clean_review)



text = data['title'][0]

def set_custom_format(text):
    words = text.split()
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
                print('it has mixed words')
        else:
            text_fixed += w

        text_fixed += ' '