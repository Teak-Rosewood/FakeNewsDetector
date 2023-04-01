import pandas as pd 
import numpy as np 
import nltk
import re

from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import preprocess_kgptalkie as ps

def initial_preprocess(x_test):
    unknown_pub = []
    for idx, row in enumerate(x_test.text.values):
        try:
            record = row.split('-', maxsplit=1)
            assert(len(record[0]) < 120)
        except:
            unknown_pub.append(idx)
    publisher = []
    tmp_text = []
    for idx, row in enumerate(x_test.text.values):
        if idx in unknown_pub:
            tmp_text.append(row)
            publisher.append('unknown')
        else:
            tmp_text.append(row)
            publisher.append('unknown')
    x_test['publisher'] = publisher
    x_test.text = tmp_text
    x_test.text = x_test.title + " " + x_test.text
    x_test.text = x_test.text.apply(lambda x:str(x).lower())
    x_test.text.apply(lambda x:ps.remove_special_chars(x))
    x_test.to_csv('data/preprocessed_data.csv')

def preprocess_data(x_test, tokenizer):
    unknown_pub = []
    for idx, row in enumerate(x_test.text.values):
        try:
            record = row.split('-', maxsplit=1)
            assert(len(record[0]) < 120)
        except:
            unknown_pub.append(idx)
    publisher = []
    tmp_text = []
    for idx, row in enumerate(x_test.text.values):
        if idx in unknown_pub:
            tmp_text.append(row)
            publisher.append('unknown')
        else:
            tmp_text.append(row)
            publisher.append('unknown')
    x_test['publisher'] = publisher
    x_test.text = tmp_text
    x_test.text = x_test.title + " " + x_test.text
    x_test.text = x_test.text.apply(lambda x:str(x).lower())
    x_test.text.apply(lambda x:ps.remove_special_chars(x))

    x_test_preprocessed = [row.split() for row in x_test.text.tolist()]
    x_test_preprocessed = tokenizer.texts_to_sequences(x_test_preprocessed)
    x_test_preprocessed = pad_sequences(x_test_preprocessed, maxlen = 1000)
    return x_test_preprocessed