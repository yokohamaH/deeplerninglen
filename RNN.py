import numpy as np
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog are my homework']

tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(samples)

sequence = tokenizer.texts_to_sequence(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print(word_index)
print(sequence)
print(one_hot_results)
print('Found %s unique tokens.' % len(word_index))
