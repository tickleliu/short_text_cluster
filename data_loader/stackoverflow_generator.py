# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: char_sen_generator.py 
@time: 2018/12/19 
"""
import os
import pickle
from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from data_loader.data_generator import DataGenerator
from utils.reduce_function import *
from utils.utils import binarize


class StackGenerator(DataGenerator):
    def __init__(self, config):
        super(StackGenerator, self).__init__(config)
        self.config = config
        self.embedding_file = config.embedding_file
        self.text_path = config.text_path

        # load traing data
        with open(self.config.text_path, encoding="utf8") as f:
            self.data = [text.strip() for text in f]

        with open(self.config.label_path) as f:
            target = f.readlines()
        self.target = np.asarray([int(label.rstrip('\n')) for label in target])
        print("Total: %s short texts" % format(len(self.data), ","))

        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(self.data)
        sequences_full = tokenizer.texts_to_sequences(self.data)
        self.tokenizer = tokenizer

        word_index = tokenizer.word_index
        self.word_index = word_index
        pickle.dump(self.word_index, open("../temp/wi.pkl", "wb"))
        print('Found %s unique tokens.' % len(word_index))
        MAX_NB_WORDS = len(word_index)

        seq_lens = [len(s) for s in sequences_full]
        print("Average length: %d" % np.mean(seq_lens))
        print("Max length: %d" % max(seq_lens))
        MAX_SEQUENCE_LENGTH = max(seq_lens)
        X = pad_sequences(sequences_full, maxlen=MAX_SEQUENCE_LENGTH)
        self.input = X
        self.config.max_seq_len = MAX_SEQUENCE_LENGTH

        # load word embedding
        print('Preparing embedding matrix')
        if os.path.exists("../temp/embedding.pkl"):
            embedding_matrix = pickle.load(open("../temp/embedding.pkl", "rb"))
        else:
            word2vec = KeyedVectors.load_word2vec_format(self.config.embedding_file, binary=True)
            nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
            embedding_matrix = np.zeros((nb_words, self.config.embedding_dim))
            for word, i in tqdm(word_index.items(), total=len(word_index.items())):
                if word in word2vec.vocab:
                    embedding_matrix[i] = word2vec.word_vec(word)
                else:
                    pass
            pickle.dump(embedding_matrix, open("../temp/embedding.pkl", "wb"))
        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        self.embedding_matrix = embedding_matrix

        # prepare model traning target
        if os.path.exists("../temp/features.pkl"):
            Y = pickle.load(open("../temp/features.pkl", "rb"))
        else:
            Y = {}
            tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
            denom = 1 + np.sum(tfidf, axis=1)[:, None]
            normed_tfidf = tfidf / denom
            average_embeddings = ae(normed_tfidf, embedding_matrix)
            print("calc ae")
            Y["ae"] = binarize(average_embeddings)
            Y["lle"] = binarize(pickle.load(open(self.config.lle_path, "rb")))
            Y["le"] = binarize(pickle.load(open(self.config.le_path, "rb")))
            Y["isomap"] = binarize(pickle.load(open(self.config.isomap_path, "rb")))
            Y["lsa"] = binarize(pickle.load(open(self.config.lsa_path, "rb")))
            Y["mds"] = binarize(pickle.load(open(self.config.mds_path, "rb")))
            pickle.dump(Y, open("../temp/features.pkl", "wb"))
        self.y = Y

    def next_batch(self, batch_size):
        size = self.input.shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)
        i = 0
        while True:
            if i + batch_size <= size:
                start_index = i
                end_index = i + batch_size
                idx = indices[start_index:end_index]
                yield self.input[idx], self.y["ae"][idx], \
                      self.y["lle"][idx], \
                      self.y["le"][idx], \
                      self.y["isomap"][idx], \
                      self.y["lsa"][idx], \
                      self.y["mds"][idx]
                i += batch_size
            else:
                i = 0
                indices = np.arange(size)
                np.random.shuffle(indices)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("embedding_file", "../data/GoogleNews-vectors-negative300.bin", "embedding file txt")
    tf.app.flags.DEFINE_string("text_path", "../data/StackOverflow.txt", "embedding file txt")
    tf.app.flags.DEFINE_string("label_path", "../data/StackOverflow_gnd.txt", "embedding file txt")
    tf.app.flags.DEFINE_string("ae_path", "../ae.pkl", "embedding file txt")
    tf.app.flags.DEFINE_string("le_path", "../le.pkl", "embedding file txt")
    tf.app.flags.DEFINE_string("lle_path", "../lle.pkl", "embedding file txt")
    tf.app.flags.DEFINE_string("isomap_path", "../isomap.pkl", "embedding file txt")
    tf.app.flags.DEFINE_string("mds_path", "../mds.pkl", "embedding file txt")
    tf.app.flags.DEFINE_string("lsa_path", "../lsa.pkl", "embedding file txt")
    tf.app.flags.DEFINE_integer("max_seq_len", 30, "seq length")
    tf.app.flags.DEFINE_integer("embedding_voc", 2000, "embedding words vocabulary")
    tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimension")
    tf.app.flags.DEFINE_integer("target_dim", 70, "word embedding dimension")
    config = tf.app.flags.FLAGS
    generator = StackGenerator(config)
    generator_next_batch = generator.next_batch(100)
    while True:
        x, y_ae, y_le, y_lle, y_isomap, y_mds, y_lsa = next(generator_next_batch)
        print(np.shape(x))
        print(np.shape(y_ae))
        print(np.shape(y_le))
        print(np.shape(y_lle))
        print(np.shape(y_isomap))
        print(np.shape(y_mds))
        print(np.shape(y_lsa))
        break
