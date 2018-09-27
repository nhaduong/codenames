from keras.layers import Input, Dense
from keras.models import Model
from codenames.codenames.embedding_handler import EmbeddingHandler
import numpy as np

eh = EmbeddingHandler('D:/Documents/mycodenames/codenames/data/uk_embeddings.txt')
def get_word_vectors(file):
    with open(file) as f:
        lines = f.readlines()
        lines = [line.split(',')[-1].strip('\n') for line in lines]
        #print(lines[:4])
        word_vectors = [eh.get_word_vector(word) for word in lines]

    return np.asarray(word_vectors)
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> from 300 original dim

# this is our input placeholder
input_clue = Input(shape=(300,))
# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_clue)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(300, activation='softmax')(decoded)

# encoded = Dense(encoding_dim, activation='relu')(input_clue)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(300, activation='softmax')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_clue, decoded)


# this model maps an input to its encoded representation
encoder = Model(input_clue, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-3](encoded_input)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)

# create the decoder model
decoder = Model(encoded_input, decoder_layer)



autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

x_train = get_word_vectors('D:/Documents/mycodenames/codenames/data/ml/train')
x_test = get_word_vectors('D:/Documents/mycodenames/codenames/data/ml/test')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# encode and decode some digits
# note that we take them from the *test* set
encoded_words = encoder.predict(x_test)
decoded_words = decoder.predict(encoded_words)

for enc, dec in zip(x_test, decoded_words):
    print(eh.get_nearest_word(enc), eh.get_nearest_word(dec))

autoencoder.save('autoencoder_model_adam.h5')