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
from keras.models import Model
from sklearn.metrics import roc_auc_score
import linecache
import pickle

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 50
NUM_SAMPLES = 100000
MAX_VOCAB_SIZE= 200

#load data

print('Loading the data...')
input_info = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair.x.info'
input = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair.x'
output_true = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair1.mt'
input_lines= []
output_lines = []
output_lines_training = []
k=0
with open(input_info,'r') as infile:
    for line in infile:

        if k>NUM_SAMPLES:
            break

        a = line.split("\t")
        if (int(a[6])> 0):
            input_lines.append(linecache.getline(input,int(a[1])+1).strip("\n"))
            k+=1

            if (int(a[5]) > 0):
                manualTranscriptions = linecache.getline(output_true, int(a[8]) + 1)
                first_mt = manualTranscriptions.split("\t")[0]
                output_lines.append(first_mt +  '>')
                output_lines_training.append('<' + first_mt)
            else:
                manualTranscriptions = linecache.getline(output_true, int(a[7]) + 1)
                first_mt = manualTranscriptions.split("\t")[0]
                output_lines.append(first_mt + '>')
                output_lines_training.append('<' + first_mt)

print("num samples:", len(input_lines))

corpus= input_lines + output_lines_training

#Tokenize the inputs
tokenizer_corpus = Tokenizer(num_words=None, char_level=True, oov_token='UNK',filters='', lower= False)
tokenizer_corpus.fit_on_texts(corpus)
input_sequences = tokenizer_corpus.texts_to_sequences(input_lines)
#get the character to index mapping for input (list the vocabulary)
word2idx_corpus = tokenizer_corpus.word_index
print('Found %s unique input tokens.' % len(word2idx_corpus))
print(word2idx_corpus)
charSet = list(word2idx_corpus.keys())
EMBEDDING_DIM = len(charSet)

with open('character_set.pickle', 'wb') as fp:
    pickle.dump(charSet, fp)
with open('character_set', 'w') as f:
    for item in charSet:
        f.write("%s\n" % item)
max_len_input = max(len(s) for s in input_sequences)

#Tokenize the output
target_sequences = tokenizer_corpus.texts_to_sequences(output_lines)
target_sequences_inputs = tokenizer_corpus.texts_to_sequences(output_lines_training)
max_len_target = max(len(s) for s in target_sequences)
max_len = max(max_len_input, max_len_target)
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer_corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
#pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len, padding = 'post')
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen= max_len,  padding = 'post')
decoder_targets = pad_sequences(target_sequences, maxlen= max_len,  padding = 'post')
# print(encoder_inputs[10])
# print(decoder_targets[10])
#character vectors
num_chars = min(MAX_VOCAB_SIZE,len(word2idx_corpus) + 1)
#charSet= "abcdefghijklmnopqrstuvwxyzABCDEDGHIJKLMNOPQRTSUVWXYZ0123456789,.<>?;':!@#$%^&*()_+-=[]{}`~"
char_to_int = dict((c,i) for i,c in enumerate(charSet))
embedding_matrix = np.zeros((num_chars, EMBEDDING_DIM))
for char, i in word2idx_corpus.items():
  if i < num_chars:
      one_idx = char_to_int.get(char)
      # words not found in embedding index will be all zeros.
      if one_idx is not None:
        embedding_matrix[i][one_idx] = 1.
#prepare embedding matrix

#will get value later

#creat embedding layer
embedding_layer= Embedding(num_chars, EMBEDDING_DIM, input_length=max_len_input, weights =[embedding_matrix])


#create targets
num_character_output = len(tokenizer_corpus.word_index)+1
decoder_targets_one_hot = np.zeros((len(input_lines), max_len ,num_character_output), dtype = 'float32')
for i,d in enumerate(decoder_targets):
    for t,c in enumerate(d):
        decoder_targets_one_hot[i,t,c] = 1


print('Building Model ...')

encoder_inputs_placeholder= Input(shape=(max_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder =LSTM(400, return_state= True, dropout = 0.5)
encoder_outputs,h,c = encoder(x)
encoder_states = [h,c]

#Setup the decoder
decoder_inputs_placeholder = Input(shape= (max_len,))
decoder_embedding= Embedding(num_character_output, 400)
decoder_inputs_x =decoder_embedding(decoder_inputs_placeholder)

#Attenion
#We are going to add attention mechanism before the decoder layer



decoder_lstm = LSTM(400, return_sequences= True, return_state= True, dropout=0.5)
decoder_outputs, _,_ = decoder_lstm(decoder_inputs_x, initial_state = encoder_states)
decoder_dense = Dense(num_character_output, activation ='softmax')
decoder_outputs =decoder_dense(decoder_outputs)

#Creat the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)
model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

fit = model.fit([encoder_inputs, decoder_inputs], decoder_targets_one_hot, batch_size=BATCH_SIZE, epochs= EPOCHS, validation_split=0.2)

# plot some data
plt.plot(fit.history['loss'], label='loss')
plt.plot(fit.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(fit.history['acc'], label='acc')
plt.plot(fit.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# Save model
model_json = model.to_json()
with open("s2smodeltest2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('s2stest2_weights.h5')
