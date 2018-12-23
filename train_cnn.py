import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle
from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from utils.utils import cluster_quality
from utils.reduce_function import *

EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin'

text_path = 'data/StackOverflow.txt'
label_path = 'data/StackOverflow_gnd.txt'

with open(text_path, encoding="utf8") as f:
    data = [text.strip() for text in f]

with open(label_path) as f:
    target = f.readlines()
target = [int(label.rstrip('\n')) for label in target]

print("Total: %s short texts" % format(len(data), ","))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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
X_ori = X
x_indices = np.arange(start=0, stop=X_ori.shape[0], step=1)
# X = X[x_indices[0:2000]]
np.random.shuffle(x_indices)
y = target

############################
# Preparing embedding matrix
############################


print('Preparing embedding matrix')
# word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

EMBEDDING_DIM = 70
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
# embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
# for word, i in word_index.items():
#     if word in word2vec.vocab:
#         embedding_matrix[i] = word2vec.word_vec(word)
#     else:
#         pass
#
# pickle.dump(embedding_matrix, open("embedding.pkl", "wb"))
embedding_matrix = pickle.load(open("temp/embedding.pkl", "rb"))
# print(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#################################################
# Preparing target using Average embeddings (AE)
#################################################

Y = {}
tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
# tfidf = tfidf[x_indices[0:2000]]
denom = 1 + np.sum(tfidf, axis=1)[:, None]
normed_tfidf = tfidf / denom
average_embeddings = np.dot(normed_tfidf, embedding_matrix)
Y["ae"] = average_embeddings

# Y["lsa"] = lsa(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["lsa"], open("lsa.pkl", "wb"))
# Y["lsa"] = pickle.load(open("lsa.pkl", "rb"))

# pickle.dump(Y["ae"], open("ae.pkl", "wb"))
# Y["lle"] = lle(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["lle"], open("lle.pkl", "wb"))
# Y["lle"] = pickle.load(open("lle.pkl", "rb"))

# normed_tfidf = pca(normed_tfidf, 1000)
# Y["le"] = le(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["le"], open("le.pkl", "wb"))
# Y["le"] = pickle.load(open("le.pkl", "rb"))

# Y["mds"] = mds(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["mds"], open("mds.pkl", "wb"))
# Y["mds"] = pickle.load(open("mds.pkl", "rb"))

# Y["tsne"] = tsne(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["tsne"], open("tsne.pkl", "wb"))
# Y["tsne"] = pickle.load(open("tsne.pkl", "rb"))

# Y["isomap"] = isomap(normed_tfidf, EMBEDDING_DIM)
# pickle.dump(Y["isomap"], open("isomap.pkl", "wb"))
# Y["isomap"] = pickle.load(open("isomap.pkl", "rb"))

print("Shape of average embedding: ", Y['ae'].shape)

# binary Y
from utils.utils import binarize

reduction_name = "ae"
# reduction_name = "lsa"
# reduction_name = "lle"
# reduction_name = "le"
# reduction_name = "mds"
# reduction_name = "tsne"
# reduction_name = "isomap"
B = binarize(Y[reduction_name])

# Last dimeaeon in the CNN
TARGET_DIM = B.shape[1]

# Example of binarized target vector
print(B.shape)
print(B[0])

################################################
# train model
################################################

from keras.layers import Input, Embedding, Flatten, Reshape
from keras.layers import Dense, Conv1D, Dropout, merge
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model


def get_model():
    embedding_matrix_copy = embedding_matrix.copy()
    trainable_embedding = False
    # Embedding ler
    pretrained_embedding_layer = Embedding(
        input_dim=nb_words,
        output_dim=300,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
    )

    # Input
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = pretrained_embedding_layer(sequence_input)

    # 1st Layer
    # x1 = Conv1D(100, 3, activation='tanh', padding='same')(embedded_sequences)
    # x2 = Conv1D(100, 4, activation='tanh', padding='same')(embedded_sequences)
    # x3 = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    # x = concatenate([x1, x2, x3])
    x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    x = Dropout(0.5)(x)
    x = Conv1D(100, 5, activation='tanh', padding='same')(x)
    # Output
    x = Dropout(0.5)(x)
    x = GlobalMaxPooling1D()(x)
    deepfeatures = Dense(480, activation="sigmoid")(x)

    # Output
    # x = Dropout(0.5)(x)
    predictions = Dense(TARGET_DIM, activation='sigmoid')(deepfeatures)
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
              epochs=nb_epoch, batch_size=200, verbose=1, shuffle=True)

    # create model that gives penultimate layer
    input = model.layers[0].input
    output = model.layers[-2].output
    model_penultimate = Model(input, output)

    # inference of penultimate layer
    H = model_penultimate.predict(X_ori)
    # H = B
    print("Sample shape: {}".format(H.shape))

    from sklearn.preprocessing import normalize
    from sklearn.cluster import KMeans

    true_labels = y
    n_clusters = len(np.unique(y))
    print("Number of classes: %d" % n_clusters)
    km = KMeans(n_clusters=n_clusters, n_jobs=1)
    result = dict()
    V = normalize(H, norm='l2')
    km.fit(V)
    pred = km.labels_
    print(pred)
    a = {'deep': cluster_quality(true_labels, pred)}
    np.save("pred.npy", pred)
    model.save_weights("model.plk")
