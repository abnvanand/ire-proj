import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputfile', '-i', required=True, default="processedData/articles-training-bypublisher.csv")
parser.add_argument('--modelsdir', '-m', required=True, default="models")
parser.add_argument('--tokenizerfile', '-t', required=True, default="tokenizer.p")
args = parser.parse_args()

INPUT_FILE = args.inputfile
MODEL_PATH = args.modelsdir
TOKENIZER_DUMP_FILE = args.tokenizerfile

# Path of fully trained models
BINARY_MODEL = f"{MODEL_PATH}/partial_lstm_binary.h5"
MULTICLASS_MODEL = f"{MODEL_PATH}/partial_lstm_multiclass.h5"

MULTITASK_MODEL = f"{MODEL_PATH}/partial_lstm_multitask.h5"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras import Sequential, Model, Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Flatten, Dense, \
    GlobalAveragePooling1D, Dropout, LSTM, CuDNNLSTM, RNN, SimpleRNN, Conv2D, GlobalMaxPooling1D
from keras import callbacks

import re

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.metrics import classification_report

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from numpy.testing import assert_allclose

"""# Preparing text data
Format text samples and labels into tensors that can be fed into a neural network.
- keras.preprocessing.text.Tokenizer
- keras.preprocessing.sequence.pad_sequences
"""

# Source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

MAX_NUM_WORDS = 50000  # dictionary size
MAX_SEQUENCE_LENGTH = 1500  # max word length of each individual article
EMBEDDING_DIM = 300  # dimensionality of the embedding vector (50, 100, 200, 300)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')


def tokenize_testdata(X):
    with open(TOKENIZER_DUMP_FILE, 'rb') as fp:
        tokenizer = pickle.load(fp)

    print(f'Found {len(tokenizer.word_index)} unique tokens.')

    sequences = tokenizer.texts_to_sequences(X)

    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return X


def reverse_to_categorical(y):
    return np.argmax(y[:5], axis=1)


"""# Load datasets"""


def perform_cleaning(text):
    text = text.lower().strip()
    text = ' '.join(e for e in text.split())
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text


df_test = pd.read_csv(filepath_or_buffer=INPUT_FILE,
                      names=['article_id', 'title', 'articleContent', 'bias', 'hyperpartisan'],
                      )
df_test['title'] = df_test['title'].fillna(value=' ')
df_test.count()

df_test['title'] = df_test['title'].map(perform_cleaning)
df_test['articleContent'] = df_test['articleContent'].map(perform_cleaning)
df_test.tail(5)

print(df_test['hyperpartisan'].value_counts())

"""# Binary classifier (Biased / Unbiased)

## Separate labels from features
"""

X_test = df_test.articleContent.values
y_test_bias = df_test.hyperpartisan.values
y_test_bias_kind = df_test.bias.values

NUM_CLASSES_BIAS = 2
NUM_CLASSES_BIAS_KIND = 5

"""## Tokenize data"""

X_test = tokenize_testdata(X_test)
y_test_bias = to_categorical(y_test_bias, num_classes=NUM_CLASSES_BIAS)

"""# Predict Using LSTM Binary"""

model = load_model(BINARY_MODEL)

"""## On test set"""

y_pred_bias = model.predict(X_test)
print(y_test_bias[:5])
print(y_pred_bias[:5])

print(classification_report(np.argmax(y_test_bias, axis=1),
                            np.argmax(y_pred_bias, axis=1),
                            target_names=['unbiased', 'biased']))

ax = plt.subplot()
cm = confusion_matrix(np.argmax(y_test_bias, axis=1), np.argmax(y_pred_bias, axis=1))
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.show()
plt.close()

"""# Predict Using LSTM multiclass (Kind of bias classifier)"""

"""## Encode labels"""

labelEncoder = LabelEncoder()
labelEncoder.fit(np.unique(y_test_bias_kind))

y_test_bias_kind = labelEncoder.transform(y_test_bias_kind)

print(y_test_bias_kind[:5])

# Inverse tranform labels

y_test_bias_kind = to_categorical(y_test_bias_kind, num_classes=NUM_CLASSES_BIAS_KIND)

"""## Split into train and validate sets"""

"""## Load Model"""

model = load_model(MULTICLASS_MODEL)

"""## On test set"""

y_pred_bias_kind = model.predict(X_test)

ax = plt.subplot()
cm = confusion_matrix(np.argmax(y_test_bias_kind, axis=1),
                      np.argmax(y_pred_bias_kind, axis=1))
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

print(classification_report(np.argmax(y_test_bias_kind, axis=1),
                            np.argmax(y_pred_bias_kind, axis=1),
                            target_names=labelEncoder.classes_))

"""```
              precision    recall  f1-score   support

       least       0.23      0.04      0.07     38296
        left       0.24      0.54      0.34     37500
 left-center       0.27      0.20      0.23     23473
       right       0.45      0.46      0.46     37500
right-center       0.03      0.01      0.01     13231

    accuracy                           0.29    150000
   macro avg       0.25      0.25      0.22    150000
weighted avg       0.28      0.29      0.25    150000
```

# Predict using Multitask model
 - task 1: biased/unbiased (binary)
 - task 2: kind of bias (multiclass)

## Load model
"""

model = load_model(MULTITASK_MODEL)

"""## Prediction on test set"""
y_pred_bias, y_pred_bias_kind = model.predict(X_test)

print(classification_report(np.argmax(y_test_bias, axis=1),
                            np.argmax(y_pred_bias, axis=1),
                            target_names=['unbiased', 'biased']))

print(classification_report(np.argmax(y_test_bias_kind, axis=1),
                            np.argmax(y_pred_bias_kind, axis=1),
                            target_names=labelEncoder.classes_))

ax = plt.subplot()
cm = confusion_matrix(np.argmax(y_test_bias, axis=1), np.argmax(y_pred_bias, axis=1))
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')

ax = plt.subplot()
cm = confusion_matrix(np.argmax(y_test_bias_kind, axis=1), np.argmax(y_pred_bias_kind, axis=1))
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
