# -*- coding: utf-8 -*-
"""ishyperpartisan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12TxneARTfcB4wmsi8N5SoueyGhb1Leh1
"""

PROJECT_DIR = "/content/drive/My Drive/ire-proj/"

import re
import numpy as np
import keras
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import pandas as pd

MAX_SEQUENCE_LENGTH = 1500  # max word length of each individual article
USE_MULTITASK_MODEL = False


def tokenize_testdata(X):
    with open(TOKENIZER_DUMP_FILE, 'rb') as fp:
        tokenizer = pickle.load(fp)

    # print(f'Found {len(tokenizer.word_index)} unique tokens.')

    sequences = tokenizer.texts_to_sequences(X)

    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return X


def perform_cleaning(text):
    text = text.lower().strip()
    text = ' '.join(e for e in text.split())
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text


MODEL_DIR = input("Enter model directory path")

BINARY_MODEL = MODEL_DIR + "/partial_lstm_binary.h5"
MULTICLASS_MODEL = MODEL_DIR + "/partial_lstm_multiclass.h5"
MULTITASK_MODEL = MODEL_DIR + "/partial_lstm_multitask.h5"

# TOKENIZER_DUMP_FILE = PROJECT_DIR+"/models/tokenizer.p"
TOKENIZER_DUMP_FILE = input("Enter full path of tokenizer dump file")

sampleFilePath = input("Enter filename containing article text")
# sampleFilePath = "sampleArticle.txt"

sfp = open(sampleFilePath)
articleContent = sfp.readlines()
articleContent = ''.join(articleContent)
df_test = pd.DataFrame([articleContent, articleContent], columns=['articleContent'])

df_test['articleContent'] = df_test['articleContent'].map(perform_cleaning)

X_test = df_test.articleContent.values

df_test.shape

X_test.shape

X_test_tokenized = tokenize_testdata(X_test)
X_test_tokenized.shape

y_pred_bias = None
y_pred_bias_kind = None

if USE_MULTITASK_MODEL:
    model = load_model(MULTITASK_MODEL)
    y_pred_bias, y_pred_bias_kind = model.predict(X_test_tokenized)
else:
    model = load_model(BINARY_MODEL)
    y_pred_bias = model.predict(X_test_tokenized)
    model = load_model(MULTICLASS_MODEL)
    y_pred_bias_kind = model.predict(X_test_tokenized)

y_pred_bias = np.argmax(y_pred_bias, axis=1)[0]

y_pred_bias_kind = np.argmax(y_pred_bias_kind, axis=1)
y_pred_bias_kind = y_pred_bias_kind[0]

if y_pred_bias == 1:
    print("The article is hyperpartisan")
else:
    print("The article is not hyperpartisan")

bias_list = ['least', 'left', 'left-center', 'right', 'right-center']
print("This article is", bias_list[y_pred_bias_kind], "biased")
