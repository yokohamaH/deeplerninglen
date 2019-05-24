from keras.datasets import imdb
from keras import preprocessing

max_frtatures = 10000

max_len = 20

# それぞれには、出現頻度上位10000の単語がひたすらlistに入れられていく　そのlistが要素になった(25000,)型のベクトル(x_train)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_frtatures)
print(x_train.shape)

# list上で作ったlistを大きさ20にして残りを削っている残す要素は後ろから20
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

print(x_train)
