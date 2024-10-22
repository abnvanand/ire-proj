import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trainingfile', '-r', required=True,
                    default="processedData/articles-training-bypublisher.csv")
parser.add_argument('--glovefile', '-r', required=True,
                    default="processedData/glove.6B.300d.txt")

args = parser.parse_args()
TRAINING_FILE = args.trainingfile
GLOVE_FILE_PATH = args.glovefile

# Commented out IPython magic to ensure Python compatibility.
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

N_TRAINING_SAMPLES = None

"""# Preparing text data
Format text samples and labels into tensors that can be fed into a neural network.
- keras.preprocessing.text.Tokenizer
- keras.preprocessing.sequence.pad_sequences
"""

# Source: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

MAX_NUM_WORDS = 50000  # dictionary size
MAX_SEQUENCE_LENGTH = 1500  # max word length of each individual article
EMBEDDING_DIM = 300  # dimensionality of the embedding vector (50, 100, 200, 300)

TOKENIZER_DUMP_FILE = 'tokenizer.p'

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')


def tokenize_trainingdata(X):
    tokenizer.fit_on_texts(X)

    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')

    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    with open(TOKENIZER_DUMP_FILE, 'wb') as fp:
        pickle.dump(tokenizer, fp)

    return X, word_index


def tokenize_testdata(X):
    with open(TOKENIZER_DUMP_FILE, 'rb') as fp:
        tokenizer = pickle.load(fp)

    print(f'Found {len(tokenizer.word_index)} unique tokens.')

    sequences = tokenizer.texts_to_sequences(X)

    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return X


def reverse_to_categorical(y):
    return np.argmax(y[:5], axis=1)


"""# Preparing the embedding layer"""


def load_embeddings(word_index):
    # Load glove word embeddings
    embeddings_index = {}
    f = open(GLOVE_FILE_PATH, 'r', encoding='utf8')
    for line in f:
        # each line starts with a word; rest of the line is the vector
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'Found {len(embeddings_index)} word vectors in glove file.')

    # Now use embedding_index dictionary and our word_index 
    # to compute our embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print("embedding_matrix shape:", np.shape(embedding_matrix))

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


"""# Load datasets"""

df = pd.read_csv(filepath_or_buffer=TRAINING_FILE,
                 names=['article_id', 'title', 'articleContent', 'bias', 'hyperpartisan'],
                 dtype={'title': str},
                 nrows=N_TRAINING_SAMPLES)

df['title'] = df['title'].fillna(value=' ')
df.count()


def perform_cleaning(text):
    text = text.lower().strip()
    text = ' '.join(e for e in text.split())
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    return text


df['title'] = df['title'].map(perform_cleaning)
df['articleContent'] = df['articleContent'].map(perform_cleaning)

df.head()
# df.tail(5)

print(df['hyperpartisan'].value_counts())

"""# Binary classifier (Biased / Unbiased)

## Separate labels from features
"""

X = df.articleContent.values
y_bias = df.hyperpartisan.values
y_bias_kind = df.bias.values

NUM_CLASSES_BIAS = len(np.unique(y_bias))
NUM_CLASSES_BIAS_KIND = len(np.unique(y_bias_kind))

print(y_bias[:5])
print(y_bias_kind[:5])

"""## Tokenize data"""

X, word_index = tokenize_trainingdata(X)
y_bias = to_categorical(y_bias, num_classes=NUM_CLASSES_BIAS)

print(y_bias[:5])
print(reverse_to_categorical(y_bias[:5]))

X_train, X_validate, y_train_bias, y_validate_bias = train_test_split(X, y_bias,
                                                                      test_size=0.2,
                                                                      random_state=12)

"""## Compile model"""

loaded_embeddings = load_embeddings(word_index)

input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = loaded_embeddings(input_layer)
embedding_layer = Dropout(0.5)(embedding_layer)

hidden_layer = LSTM(64, recurrent_dropout=0.5)(embedding_layer)
hidden_layer = Dropout(0.5)(hidden_layer)
output_layer = Dense(2, activation='softmax')(hidden_layer)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

print(model.summary())

"""## Fit model"""

CHECKPOINT_FILE = 'partial_lstm_binary.h5'

checkpoint = ModelCheckpoint(CHECKPOINT_FILE, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model = None

model.fit(X_train, y_train_bias,
          validation_data=(X_validate, y_validate_bias),
          epochs=25, batch_size=250,
          callbacks=callbacks_list)

# new_model = load_model(CHECKPOINT_FILE)

y_pred_bias_validate = model.predict(X_validate)
print(classification_report(np.argmax(y_validate_bias, axis=1),
                            np.argmax(y_pred_bias_validate, axis=1),
                            target_names=['unbiased', 'biased']))

"""## Save model"""

model.save('lstm_binary.h5')

"""# Multiclass classifier (Kind of bias classifier)

## Separate labels from features
"""

print(y_bias_kind[:5])

"""## Encode labels"""

labelEncoder = LabelEncoder()
labelEncoder.fit(np.unique(y_bias_kind))

y_bias_kind = labelEncoder.transform(y_bias_kind)

print(y_bias_kind[:5])

# Inverse tranform labels
labelEncoder.inverse_transform(y_bias_kind[:5])

y_bias_kind = to_categorical(y_bias_kind, num_classes=NUM_CLASSES_BIAS_KIND)

# TO get Reverse of to_categorical
print(reverse_to_categorical(y_bias_kind))

"""## Split into train and validate sets"""

X_train, X_validate, y_train_bias_kind, y_validate_bias_kind = train_test_split(X,
                                                                                y_bias_kind,
                                                                                test_size=0.2,
                                                                                random_state=12)

"""## Compile model"""

input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = loaded_embeddings(input_layer)
embedding_layer = Dropout(0.5)(embedding_layer)

hidden_layer = LSTM(64, recurrent_dropout=0.5)(embedding_layer)
hidden_layer = Dropout(0.5)(hidden_layer)
output_layer = Dense(NUM_CLASSES_BIAS_KIND, activation='softmax')(hidden_layer)

model = Model(input_layer, output_layer)
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

print(model.summary())

"""## Fit model"""

CHECKPOINT_FILE = 'partial_lstm_multiclass.h5'

checkpoint = ModelCheckpoint(CHECKPOINT_FILE, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, y_train_bias_kind,
          validation_data=(X_validate, y_validate_bias_kind),
          epochs=25, batch_size=250,
          callbacks=callbacks_list)

# new_model = load_model(CHECKPOINT_FILE)

y_pred_bias_kind_validate = model.predict(X_validate)
print(classification_report(np.argmax(y_validate_bias_kind, axis=1),
                            np.argmax(y_pred_bias_kind_validate, axis=1),
                            target_names=labelEncoder.classes_))

"""## Save model"""

model.save('lstm_multiclass.h5')

"""# Multitask learning 
 - task 1: biased/unbiased (binary)
 - task 2: kind of bias (multiclass)
"""

input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = loaded_embeddings(input_layer)
embedding_layer = Dropout(0.5)(embedding_layer)

hidden_layer = LSTM(64, recurrent_dropout=0.5)(embedding_layer)
hidden_layer = Dropout(0.5)(hidden_layer)

# task 1
output_bias = Dense(2, activation='softmax')(hidden_layer)

# task 2
output_bias_kind = Dense(5, activation='softmax')(hidden_layer)

model = Model(input_layer, [output_bias, output_bias_kind])

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['acc'])

print(model.summary())

"""## Fit model"""

CHECKPOINT_FILE = 'partial_lstm_multitask.h5'

checkpoint = ModelCheckpoint(CHECKPOINT_FILE, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X_train, [y_train_bias, y_train_bias_kind],
          validation_data=(X_validate, [y_validate_bias, y_validate_bias_kind]),
          epochs=25, batch_size=250,
          callbacks=callbacks_list)

# model = load_model(CHECKPOINT_FILE)

y_pred_bias_validate, y_pred_bias_kind_validate = model.predict(X_validate)

print(classification_report(np.argmax(y_validate_bias, axis=1),
                            np.argmax(y_pred_bias_validate, axis=1),
                            target_names=['unbiased', 'biased']))

print(classification_report(np.argmax(y_validate_bias_kind, axis=1),
                            np.argmax(y_pred_bias_kind_validate, axis=1),
                            target_names=labelEncoder.classes_))

"""## Save model"""

model.save('lstm_multitask.h5')
