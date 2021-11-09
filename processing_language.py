import pickle
import os
from pyvi import ViTokenizer
import string
from tqdm import tqdm
from underthesea import word_tokenize
import keras

file_list = []
for (dirpath, dirname, filename) in os.walk('data/'):
    for f in filename:
        file_list.append(dirpath+'/'+f)
print(len(file_list))

def clean_document(doc):
    doc = ViTokenizer.tokenize(doc)
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

INPUT_LENGTH = 50
sequences = []
count = 0
for f in tqdm(file_list):
    f1 = open(f, encoding='utf-16')
    doc = f1.read()
    tokens = clean_document(doc)
    for i in range(INPUT_LENGTH + 1, len(tokens)):
        seq = tokens[i-INPUT_LENGTH-1:i]
        line = ' '.join(seq)
        sequences.append(line)

tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ')
tokenizer.fit_on_texts(sequences)
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
sequences_digit = tokenizer.texts_to_sequences(sequences)
pickle.dump(sequences_digit, open("sequences_digit.pkl", "wb"))