import json
import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv1D, Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
from utils.reduce_function import *

from utils.utils import binarize

EMBEDDING_FILE = 'data/word_embedding.txt'
text_path = 'data/sens.npy'

with open(text_path, "rb") as f:
    sens = pickle.load(f)
data = []
for sen in sens:
    line = []
    for word in sen:
        line.extend([char for char in word])
    line = " ".join(line)
    data.append(line)
# data = data[0:1000]

print("Total: %s short texts" % format(len(data), ","))

tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(data)
sequences_full = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
MAX_NB_WORDS = len(word_index)

seq_lens = [len(s) for s in sequences_full]
print("Average length: %d" % np.mean(seq_lens))
print("Max length: %d" % max(seq_lens))
MAX_SEQUENCE_LENGTH = max(seq_lens)

X = pad_sequences(sequences_full, maxlen=MAX_SEQUENCE_LENGTH)

############################
# Preparing embedding matrix
############################


print('Preparing embedding matrix')

EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
word2vec = {}

topn = 100000
lines_num = 0
with open(EMBEDDING_FILE, encoding='utf-8', errors='ignore') as f:
    first_line = True
    for line in tqdm(f, total=topn):
        if first_line:
            first_line = False
            dim = int(line.rstrip().split()[1])
            continue
        tokens = line.rstrip().split(' ')
        word2vec[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
        lines_num += 1
        if topn != 0 and lines_num >= topn:
            break

for word, i in word_index.items():
    if word in word2vec:
        embedding_matrix[i] = word2vec[word]
    else:
        try:
            print(word)
        except:
            pass
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#################################################
# Preparing target using Average embeddings (AE)
#################################################
Y = {}
tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
denom = 1 + np.sum(tfidf, axis=1)[:, None]
normed_tfidf = tfidf / denom

# save temp file
pickle.dump(embedding_matrix, open("temp/embedding.npy", "wb"))
pickle.dump(normed_tfidf, open("temp/features.npy", "wb"))

embedding_matrix = pickle.load(open("temp/embedding.npy", "rb"))
normed_tfidf = pickle.load(open("temp/features.npy", "rb"))

# average_embeddings = np.dot(normed_tfidf, embedding_matrix)
average_embeddings = ae(normed_tfidf, embedding_matrix)
Y["ae"] = average_embeddings
print("Shape of average embedding: ", Y['ae'].shape)
#
Y["lle"] = lle(normed_tfidf, 300)

# binary Y
reduction_name = "ae"
reduction_name = "lle"
B = binarize(Y[reduction_name])

# Last dimension in the CNN
TARGET_DIM = B.shape[1]

# Example of binarized target vector
print(B.shape)
print(B[0])


################################################
# train model
################################################


def get_model():
    embedding_matrix_copy = embedding_matrix.copy()
    trainable_embedding = False
    # Embedding layer
    pretrained_embedding_layer = Embedding(
        input_dim=nb_words,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
    )

    # Input
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = pretrained_embedding_layer(sequence_input)

    # 1st Layer
    x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)

    # Output
    x = Dropout(0.5)(x)
    predictions = Dense(TARGET_DIM, activation='sigmoid')(x)
    model = Model(sequence_input, predictions)

    model.layers[1].trainable = trainable_embedding

    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Loss and Optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['mae'])
    # Fine-tune embeddings or not
    model.summary()
    return model


if __name__ == '__main__':
    nb_epoch = 1
    checkpoint = ModelCheckpoint('models/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='auto')
    model = get_model()
    model.fit(X, B, validation_split=0.2,
              epochs=nb_epoch, batch_size=100, verbose=1, shuffle=True)

    # create model that gives penultimate layer
    input = model.layers[0].input
    output = model.layers[-2].output
    model_penultimate = Model(input, output)

    # inference of penultimate layer
    H = model_penultimate.predict(X)
    print("Sample shape: {}".format(H.shape))

    # true_labels = y
    # n_clusters = len(np.unique(y))
    n_clusters = 1000
    print("Number of classes: %d" % n_clusters)
    km = KMeans(n_clusters=n_clusters, n_jobs=10)
    result = dict()
    V = normalize(H, norm='l2')
    km.fit(V)
    pred = km.labels_
    print(pred)
    # a = {'deep': cluster_quality(true_labels, pred)}
    with open("result.json", "w") as f:
        class_dict = {}
        for index, class_num in enumerate(pred):
            class_sens = class_dict.get(class_num, [])
            class_sen = data[index]  # type:str
            class_sen = class_sen.replace(" ", "")
            class_sens.append(class_sen)
            class_dict[class_num] = class_sens
        for key, value in class_dict.items():
            f.writelines(json.dumps({str(key): value}, ensure_ascii=False) + "\n")
        np.save("temp/pred.npy", pred)
        model.save_weights("temp/model.plk")
