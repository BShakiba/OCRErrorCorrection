from __future__ import print_function, division
from builtins import range

from numpy.random import seed
seed(1)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, GRU, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, RepeatVector
from keras.models import Model, load_model, model_from_json
from sklearn.metrics import roc_auc_score
import linecache
from keras.utils.vis_utils import plot_model
import pickle

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 40
NUM_SAMPLES = 100000
LATENT_DIM = 400
max_len = 125
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Tokenize the inputs

json_file = open('s2smodeltest.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("s2stest_weights.h5",  by_name=True)
print("Loaded model from disk")
loaded_model.summary()

# encoder_model = Model(encoder_inputs_placeholder, encoder_states)
encoder_inputs_placeholder= Input(shape=(max_len,))
embedding_layer = loaded_model.get_layer(name= 'embedding_1')
encoder= loaded_model.get_layer(name = 'lstm_1')
x = embedding_layer(encoder_inputs_placeholder)
encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]
encoder_model = Model(encoder_inputs_placeholder, encoder_states)


decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_embedding = loaded_model.get_layer(name= 'embedding_2')
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
decoder_lstm = loaded_model.get_layer(name = 'lstm_2')
decoder_outputs, h, c = decoder_lstm(
  decoder_inputs_single_x,
  initial_state=decoder_states_inputs
)

decoder_states = [h, c]
decoder_dense = loaded_model.get_layer(name='dense_1')
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
  [decoder_inputs_single] + decoder_states_inputs,
  [decoder_outputs] + decoder_states
)
word2idx = tokenizer.word_index
idx2word_in = {v:k for k, v in word2idx.items()}
idx2word_trans = {v:k for k, v in word2idx.items()}
print ('test')




input_test=['the dorldd']
#Tokenize the inputs
input_sequences = tokenizer.texts_to_sequences(input_test)
#get the character to index mapping for input (list the vocabulary)
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len)
states_value = encoder_model.predict(encoder_inputs)
target_seq = np.zeros((1, 1))
target_seq[0, 0] = word2idx['<']
eos = word2idx['>']
output_sentence = []
for _ in range(max_len):
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    idx = np.argmax(output_tokens[0, 0, :])
    if eos == idx:
        break
    word = ''
    if idx>0:
        word = idx2word_trans[idx]
        output_sentence.append(word)
    target_seq[0,0] = idx
    states_value = [h,c]
print(' '.join(output_sentence))

