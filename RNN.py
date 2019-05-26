from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

max_frtatures = 10000

max_len = 20

# それぞれには、出現頻度上位10000の単語がひたすらlistに入れられていく　そのlistが要素になった(25000,)型のベクトル(x_train)
# y_trainは正解ラベル
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_frtatures)

# list上で作ったlistを大きさ20にして残りを削っている残す要素は後ろから20
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

model = Sequential()

model.add(Embedding(10000, 8, input_length=max_len))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_split=0.2)
