import pandas as pd
import numpy as np
import keras
import re
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adagrad
import tensorflow as tf
import keras.backend as K
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

if os.path.exists("/data"):
    data_path = "/data"
else:
    data_path = "data"

if os.path.exists("/output"):
    output_path = "/output"
else:
    output_path = "output"

if os.path.exists("/glove"):
    glove_path = "/glove"
else:
    glove_path = "glove"


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

df_train = pd.read_csv(data_path + '/train.csv')
df_test = pd.read_csv(data_path + '/test.csv')

words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove_path + '/glove.6B.50d.txt')

word_count = df_train['comment_text'].map(lambda text : len(text.split()))


def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))

    for i in range(m):
        sentence_words = X[i].lower().split()

        j = 0

        for j in range(min(len(sentence_words), max_len)):
            w = sentence_words[j]
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]

    return X_indices

X_train = sentences_to_indices(df_train['comment_text'].values, words_to_index, 100)

Y_train = df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

X_test = sentences_to_indices(df_test['comment_text'].values, words_to_index, 100)

X_train.shape

Y_train.shape

emb_dim = 50
vocab_len = len(words_to_index) + 1
emb_matrix = np.zeros((vocab_len, emb_dim))

for word, index in words_to_index.items():
    emb_matrix[index, :] = word_to_vec_map[word]

embedding = Embedding(vocab_len, emb_dim, trainable=False, weights=[emb_matrix])


def get_model(input_shape):
    x_input = Input(shape=input_shape, dtype='int32')

    x = embedding(x_input)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=x_input, outputs=x)
    return model


model = get_model(input_shape=(100,))
model.summary()

def auc(y_true, y_pred):
    value, update_op = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
    return value
model.compile(loss='binary_crossentropy', metrics=[auc, 'accuracy'], optimizer=Adagrad(0.1))

file_path = output_path + "/weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]
model.fit(X_train, Y_train, validation_split=0.1, epochs=2, batch_size=128, callbacks=callbacks)

embedding.trainable = True

model.fit(X_train, Y_train, validation_split=0.1, epochs=2, batch_size=128, callbacks=callbacks)

model = load_model(file_path, custom_objects={'auc':auc})

print("predicting.....")
Y_test = model.predict(X_test, verbose=1)

df_submission = pd.DataFrame()
df_submission['id'] = df_test['id']
for i, column in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
    df_submission[column] = Y_test[:, i]
df_submission.to_csv(output_path + '/submission.csv', index=False)