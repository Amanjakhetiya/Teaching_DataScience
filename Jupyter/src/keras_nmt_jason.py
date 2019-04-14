# Ref : https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

num_encoder_tokens = 71 # number of distinct characters in English, dimension of one hot encoded vector
num_decoder_tokens = 93 # number of distinct characters in French, dimension of one hot encoded vector
latent_dim = 256 # dim of context vector

# define network
encoder_inputs = Input(shape=(None,num_encoder_tokens))
encoder = LSTM(latent_dim,return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# not interested in encoder outputs, keeping rest
encoder_states = [state_h,state_c]

decoder_inputs = Input(shape=(None,num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

plot_model(model, to_file='models/nmt.png', show_shapes=True)

# for inferencing
encoder_model = Model(encoder_inputs,encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# summarize model
plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)



