from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, GRU, Bidirectional
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, RepeatVector
from keras.models import Model
from sklearn.metrics import roc_auc_score

max_len_input = 1000
max_len_target = 1000
EMBEDDING_DIM = 10
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10



#load data

print('Loading the data...')
input_texts = []
target_texts = []
target_texts_inputs = []#%Sentence in target language
t=0
for line in open('../data.txt'):
    t+=1
    if '\t' not in line:
        continue
    input_text, corrected_text = line.split('\t')
    target_texts = corrected_text + '>'
    target_texts_inputs = '<'+ corrected_text
    input_texts.append(input_text)
    target_texts.append(target_texts)
    target_texts_inputs.append(target_texts_inputs)

print("num samples:", len(input_texts))



#Tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)
#get the character to index mapping for input (list the vocabulary)
print(tokenizer_inputs.word_index)



#Tokenize the output
tokenizer_outputs = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)
#get the word to index mapping for the output
print(tokenizer_outputs.word_index)

#pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen= max_len_target, padding = 'post')
decoder_targets =pad_sequences(target_sequences, maxlen= max_len_target, padding = 'post')

#loading pretrained character vectors


#prepare embedding matrix
embedding_matrix= []
#will get value later

#creat embedding layer
vocab_size = len(tokenizer_inputs.word_index)
embedding_layer= Embedding(vocab_size+1, EMBEDDING_DIM, input_length=max_len_input, weights =[embedding_matrix])


#create targets
num_character_output = len(tokenizer_outputs.word_index)+1
decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target,num_character_output), dtype = 'float32')
for i,d in enumerate(decoder_targets):
    for t,c in enumerate(d):
        decoder_targets_one_hot[i,t,c] = 1


print('Building Model ...')

encoder_inputs_placeholder= Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(10, return_state= True, dropout = 0.5))
encoder_outputs,h,c = encoder(x)
encoder_states = [h,c]

#Setup the decoder
decoder_inputs_placeholder = Input(shape= (max_len_target,))
decoder_embedding= Embedding(num_character_output, 10)
decoder_inputs_x =decoder_embedding(decoder_inputs_placeholder)

#Attenion
#We are going to add attention mechanism before the decoder layer



decoder_lstm = LSTM(10, return_sequences= True, return_state= True, dropout=0.5)
decoder_outputs, _,_ = decoder_lstm(decoder_inputs_x, initial_state = encoder_states)
decoder_dense = Dense(num_character_output, activation ='softmax')
decoder_outputs =decoder_dense(decoder_outputs)

#Creat the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(optimizer = 'rmsprop', loss='catagorical_crossentropy', metrics=['accuarcy'])

fit = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot, batch_size=BATCH_SIZE, epochs= EPOCHS, validation_split=0.2)



