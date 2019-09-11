import re
import string


def has_numbers(input_string):
    """ test if the string is a number"""
    return bool(re.search(r'\d', input_string))


def custom_text_format(text_arg):
    """Apply a custom cleaning process: remove special characters,
    and set numbers and numbers-words with tags"""
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
                    text_fixed += 'isanumber'
                    # text_fixed += ''
                except:
                    text_fixed += 'isamixed'
                    # print('it has mixed words')
                    # text_fixed += ''
            else:
                text_fixed += w


        else:
            text_fixed += ''

        text_fixed += ' '
    # text_fixed = re.sub("[!@#$%^&*()[]{};:,./<>?\|`~-=_+]", " ", text_fixed)

    return text_fixed.lower()
