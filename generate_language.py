import pickle
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from processing_language import clean_document
import string

model = load_model('language_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('sequences_digit.pkl', 'rb') as f:
    sequences_digit = pickle.load(f)

def preprocess_input(doc):
    tokens = clean_document(doc)
    tokens = tokenizer.texts_to_sequences(tokens)
    tokens = pad_sequences([tokens], maxlen=50, truncating='pre')
    return np.reshape(tokens, (1,50))

def generate_text(text_input, n_words):
    tokens = preprocess_input(text_input)
    for _ in range(n_words):
        next_digit = model.predict_classes(tokens)
        tokens = np.append(tokens, next_digit)
        tokens = np.delete(tokens, 0)
        tokens = np.reshape(tokens, (1, 50))
    
    # Mapping to text  
    tokens = np.reshape(tokens, (50))
    out_word = []
    for token in tokens:
        for word, index in tokenizer.word_index.items():
            if index == token:
                out_word.append(word)
                break

    return ' '.join(out_word)