# Building a chatbot with Deep NLP 
# (Ref: https://www.udemy.com/chatbot/learn/v4/t/lecture/8789938?start=0)

import numpy as np
import tensorflow as tf
import re
import time

# Importing the dataset
lines = open("data/movie_lines.txt",encoding="utf-8",errors="ignore").read().split("\n")
conversations = open("data/movie_conversations.txt",encoding="utf-8",errors="ignore").read().split("\n")

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    words = line.split(" +++$+++ ")
    if len(words) == 5:
        id2line[words[0]] = words[4]
        
# Creating a list of all conversations
conversations_ids = []
for conversation in conversations[:-1]: # avoiding the last row which is empty
    cleaned_last_column = conversation.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(cleaned_last_column.split(","))
    
# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Doing a first cleanup of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)    
    text = re.sub(r"where's","where is",text)        
    text = re.sub(r"\'ll"," will",text)         
    text = re.sub(r"\'ve"," have",text) 
    text = re.sub(r"\'re"," are",text)     
    text = re.sub(r"\'d"," would",text)      
    text = re.sub(r"don't","do not",text)
    text = re.sub(r"can't","cannot",text)    
    text = re.sub(r"won't","would not",text)        
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
    return text

clean_questions = [clean_text(q) for q in questions]
clean_answers = [clean_text(a) for a in answers]

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
threshold = 20
questions_words2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questions_words2int[word] = word_number
        word_number += 1
        
answers_words2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answers_words2int[word] = word_number
        word_number += 1

tokens = ['<PAD>','<EOS>','<OUT>','<START>']    
for token in tokens:
    questions_words2int[token] = len(questions_words2int) + 1
    answers_words2int[token] = len(answers_words2int) + 1

questions_int2word = {w_i: w for w, w_i in questions_words2int.items()}
answers_int2word = {w_i: w for w, w_i in answers_words2int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"
    
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question:
        if word not in questions_words2int:
            ints.append(questions_words2int['<OUT>'])
        else:
            ints.append(questions_words2int[word])
    questions_to_int.append(ints)
        
answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer:
        if word not in answers_words2int:
            ints.append(answers_words2int['<OUT>'])
        else:
            ints.append(answers_words2int[word])
    answers_to_int.append(ints)    
    
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
def model_inputs():
    inputs = tf.placeholder(tf.int32,[None,None], name = "input")
    targets = tf.placeholder(tf.int32,[None,None], name = "target")
    lr = tf.placeholder(tf.float32,name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32,name = "dropout_rate")
    return inputs, targets, lr, keep_prob

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1],word2int["<START>"]) # One vertical column of START tokens is created
    right_side = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1]) # Matrix of targets
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    return preprocessed_targets

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = \
                        tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
    
def decoder_rnn(decoder_embedded_input,decoder_embeddings_matrix,encoder_state,
                num_words, sequence_length, rnn_size, num_layers, 
                word2int, keep_prob,batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                    num_words,None,scope=decoding_scope,
                    weights_initializer=weights,biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,decoder_cell,
            decoder_embeddings_matrix,word2int["<START>"],word2int["<EOS>"],
            sequence_length -1 ,num_words,decoding_scope, output_function, keep_prob, batch_size)
        
        return training_predictions, test_predictions
    
def seq2seq_model(inputs, targets, keep_prob, batch_size,sequence_length,
                  answers_num_words, questions_num_words, 
                  encoder_embedding_size, decoder_embedding_size,
                  rnn_size, num_layers,questions_words2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                answers_num_words + 1, encoder_embedding_size,
                initializer=tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size,
                num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_words2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets )
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                            decoder_embeddings_matrix,encoder_state,
                            questions_num_words, sequence_length, rnn_size, num_layers, 
                            questions_words2int, keep_prob,batch_size)
    return training_predictions, test_predictions

# Setting the hyperparamaters
epochs = 5 #100
batch_size = 128 #64
rnn_size = 128 # 512
num_layers = 3
encoding_embedding_size = 128 # 512
decoding_embedding_size = 128 # 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

tf.reset_default_graph()
session = tf.InteractiveSession()

inputs, targets, lr, keep_prob = model_inputs()
sequence_length = tf.placeholder_with_default(25,None,name="sequence_length")
input_shape = tf.shape(inputs)

training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]), 
                targets, keep_prob, batch_size,sequence_length,
                  len(answers_words2int), len(questions_words2int), 
                  encoding_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers,questions_words2int)

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable) for grad_tensor, grad_variable in gradients]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)