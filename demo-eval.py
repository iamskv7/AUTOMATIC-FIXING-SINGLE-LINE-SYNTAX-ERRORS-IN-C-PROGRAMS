import tensorflow as tf 
import numpy as np
import pandas as pd
import ast
from tensorflow import keras 
import matplotlib.pyplot as pyplot
from keras.layers import Embedding
from keras.utils.np_utils import to_categorical 
from google.colab import drive
from sklearn.preprocessing import OneHotEncoder
import sys

batch_size = 64  # Batch size for training.
epochs = 64  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 14643  # Number of samples to train on.
max_sequence_length = 30   #length of decoder and encoder sentence length
k = 250  # no of tokens to consider 


#loading training data 

df = pd.read_csv('/content/gdrive/MyDrive/ASEML/train.csv')
data_train = pd.DataFrame(df,columns=['sourceLineTokens','targetLineTokens'])


#applying ast 
for i in range(len(df)):
    data_train['sourceLineTokens'][i] = ast.literal_eval(df['sourceLineTokens'][i])
for i in range(len(df)):
    data_train['targetLineTokens'][i] = ast.literal_eval(df['targetLineTokens'][i])




#Loading the testing data 


df = pd.read_csv('/content/gdrive/MyDrive/ASEML/'+sys.argv[1])
data_test = pd.DataFrame(df,columns=['sourceLineTokens','targetLineTokens'])

print(len(df))
#applying ast 
for i in range(len(df)):
    data_test['sourceLineTokens'][i] = ast.literal_eval(df['sourceLineTokens'][i])
for i in range(len(df)):
    data_test['targetLineTokens'][i] = ast.literal_eval(df['targetLineTokens'][i])



class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token

    def __init__(self):
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]





# Preparing the train data and Dictionary.

v_s = Vocabulary()
v_t = Vocabulary()





dict_source = []
dict_target = []
count = 0



for item in data_train['sourceLineTokens']:
    
     item.insert(0,"SOS")
     item.append("EOS")
    
     n=len(item)
     while(n<max_sequence_length):
       item.append("PAD")
       n+=1



for item in data_train['targetLineTokens']:
    
    item.insert(0,"SOS")
    item.append("EOS")
  
    n=len(item)
    while(n<max_sequence_length):
      item.append("PAD")
      n+=1

for i in range(len(data_train)):
    v_s.add_sentence(data_train['sourceLineTokens'][i])
    v_t.add_sentence(data_train['targetLineTokens'][i])


      

dict_source_count = []
dict_target_count = []

dict_source_count = v_s.word2count
dict_target_count = v_t.word2count 




dict_source_count = sorted(dict_source_count.items(), key=lambda x: x[1],reverse = True)
dict_target_count = sorted(dict_target_count.items(), key=lambda x: x[1],reverse = True)


# making dictionary with top k tokens 
for i in range(k):
    dict_source.append(dict_source_count[i][0])
    dict_target.append(dict_target_count[i][0])



dict_source = {key: i for i, key in enumerate(dict_source)}  
dict_target = {key: i for i, key in enumerate(dict_target)}






#Adding "OOV" into dictionary 

input_token_index = dict([(item[0], i) for i, item in enumerate(dict_source.items())])  #dictionary for input_token_index
input_token_index["OOV"] = k
target_token_index = dict([(item[0], i) for i, item in enumerate(dict_target.items())])   #dictionary for output_token_index
target_token_index["OOV"] = k

# Preparing the test data.

for item in data_test['sourceLineTokens']:
    
     item.insert(0,"SOS")
     item.append("EOS")
    
     n=len(item)
     while(n<max_sequence_length):
       item.append("PAD")
       n+=1



for item in data_test['targetLineTokens']:
    
    item.insert(0,"SOS")
    item.append("EOS")
  
    n=len(item)
    while(n<max_sequence_length):
      item.append("PAD")
      n+=1

      
#encoding for test 


test_source_data_onehot = []
test_target_data_onehot = []

for src in data_test['sourceLineTokens']:
  
  lst = []
  for item2 in src:
    if input_token_index.get(item2) == None:
      lst.append(k)
    else:
      lst.append(input_token_index[item2])

  test_source_data_onehot.append(lst)    


for tgt in data_test['targetLineTokens']:
  
  
  lst = []
  for item2 in tgt:
    if target_token_index.get(item2) == None:
      lst.append(k)
    else:
      lst.append(target_token_index[item2])
  
  test_target_data_onehot.append(lst)

#padding and truncating  test data 

test_source_data_onehot = keras.preprocessing.sequence.pad_sequences(test_source_data_onehot,maxlen = max_sequence_length,padding= 'post',truncating= 'post')
test_target_data_onehot = keras.preprocessing.sequence.pad_sequences(test_target_data_onehot,maxlen = max_sequence_length,padding= 'post',truncating= 'post')


#One Hot Encoding for Test
#Encoder Input Data




test_encoder_input_data = np.zeros(
    (len(data_test), max_sequence_length, k+1), dtype="float32"
)

for i in range(len(test_source_data_onehot)):
  for j in range(len(test_source_data_onehot[i])):
    kj = test_source_data_onehot[i][j]
    test_encoder_input_data[i][j][kj] = 1.0


# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("/content/gdrive/MyDrive/ASEML/model")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="input_4t")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, k+1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["SOS"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "EOS" or len(decoded_sentence) > max_sequence_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, k+1))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence



input_texts = data_test['sourceLineTokens']
output = []

for seq_index in range(len(data_test)):
  input_seq = test_encoder_input_data[seq_index : seq_index + 1]
  decoded_sentence = decode_sequence(input_seq)

  
  output_check = []
  for i in range(len(decoded_sentence)):
    if decoded_sentence[i] not in ['SOS', 'EOS', 'PAD']:
      output_check.append(decoded_sentence[i])

  
  output.append(output_check)

#saving into csv

data_test = pd.read_csv('/content/gdrive/MyDrive/ASEML/'+sys.argv[1])
data_test["fixedTokens"] = output
data_test.to_csv("/content/gdrive/MyDrive/ASEML/"+sys.argv[2])