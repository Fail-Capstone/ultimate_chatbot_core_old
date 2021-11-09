import pickle
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, GRU, Dense, Input, Add, Concatenate, Reshape, Lambda, BatchNormalization, Dropout, Embedding, LSTM
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import utils as np_utils
# sequences_digit = tokenizer.texts_to_sequences(sequences)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

def data_generator(sequences_digit, batch_size):
    while True:
        batch_paths = np.random.choice(a = len(sequences_digit), size = batch_size)
        input = []
        output = []
        for i in batch_paths:
            input.append(sequences_digit[i][:-1])
            output.append(sequences_digit[i][-1])
        input = np.array(input)
        output = keras.utils.np_utils.to_categorical(output, num_classes=vocab_size)
        output = np.array(output)
        yield (input, output)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=50))
model.add(BatchNormalization())
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath = 'weights-training-improvement-languagemodel.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
batch_size=512
model.fit_generator(generator=data_generator(sequences_digit, batch_size), steps_per_epoch=(len(sequences_digit)//batch_size) , epochs=100, verbose=1, callbacks=callbacks_list)

model.save('language_model.h5')