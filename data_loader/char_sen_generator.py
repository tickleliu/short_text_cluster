# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: char_sen_generator.py 
@time: 2018/12/19 
"""
import os
import pickle

import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from data_loader.data_generator import DataGenerator
from utils.reduce_function import *
from utils.utils import binarize
import random


class CharSenGenerator(DataGenerator):

    def __init__(self, config):
        super(CharSenGenerator, self).__init__(config)
        self.config = config
        self.embedding_file = config.embedding_file
        self.text_path = config.text_path

        # load traing data
        if os.path.exists("../temp/sen_data.pkl"):
            self.data= pickle.load(open("../temp/sen_data.pkl", "rb"))
        else:
            with open(self.config.text_path, encoding="utf8") as f:
                self.data = [text.strip() for text in f]
                random.shuffle(self.data)
                pickle.dump(self.data, open("../temp/sen_data.pkl", "wb"))
        tokenizer = Tokenizer(char_level=False)
        self.tokenizer = tokenizer
        tokenizer.fit_on_texts(self.data)
        sequences_full = tokenizer.texts_to_sequences(self.data)
        word_index = tokenizer.word_index
        pickle.dump(word_index, open("../temp/sen_wi.pkl", "wb"))
        print('Found %s unique tokens.' % len(word_index))
        # MAX_NB_WORDS = len(word_index)

        print('Preparing embedding matrix')
        if os.path.exists("../temp/sen_embedding.pkl"):
            embedding_matrix = pickle.load(open("../temp/sen_embedding.pkl", "rb"))
        else:
            nb_words = min(self.config.voc_size, len(word_index)) + 1
            embedding_matrix = np.zeros((nb_words, config.embedding_dim))
            word2vec = {}

            topn = config.voc_size
            lines_num = 0
            with open(self.embedding_file, encoding='utf-8', errors='ignore') as f:
                first_line = True
                for line in tqdm(f):
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
            pickle.dump(embedding_matrix, open("../temp/sen_embedding.pkl", "wb"))
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

        # indexs = np.arange(0, len(sequences_full), step=1)
        # np.random.shuffle(indexs)
        sequences_full = sequences_full[0:self.config.sample_num]
        self.input = pad_sequences(sequences_full, maxlen=config.max_seq_len)

        if os.path.exists("../temp/sen_features.pkl"):
            Y = pickle.load(open("../temp/sen_features.pkl", "rb"))
        else:
            Y = {}
            tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
            denom = 1 + np.sum(tfidf, axis=1)[:, None]
            normed_tfidf = tfidf / denom
            average_embeddings = ae(normed_tfidf, embedding_matrix)
            print("calc ae")
            Y["ae"] = binarize(average_embeddings)

            print("calc lle")
            if os.path.exists(self.config.lle_path):
                Y["lle"] = pickle.load(open(self.config.lle_path, "rb"))
            else:
                Y["lle"] = binarize(lle(normed_tfidf, 70))
                pickle.dump(Y["lle"], open(self.config.lle_path, "wb"))

            print("calc le")
            if os.path.exists(self.config.le_path):
                Y["le"] = pickle.load(open(self.config.le_path, "rb"))
            else:
                Y["le"] = binarize(le(normed_tfidf, 70))
                pickle.dump(Y["le"], open(self.config.le_path, "wb"))

            print("calc isomap")
            if os.path.exists(self.config.isomap_path):
                Y["isomap"] = pickle.load(open(self.config.isomap_path, "rb"))
            else:
                Y["isomap"] = binarize(isomap(normed_tfidf, 70))
                pickle.dump(Y["isomap"], open(self.config.isomap_path, "wb"))

            print("calc lsa")
            if os.path.exists(self.config.lsa_path):
                Y["lsa"] = pickle.load(open(self.config.lsa_path, "rb"))
            else:
                Y["lsa"] = binarize(lsa(normed_tfidf, 70))
                pickle.dump(Y["lsa"], open(self.config.lsa_path, "wb"))

            print("calc mds")
            if os.path.exists(self.config.mds_path):
                Y["mds"] = pickle.load(open(self.config.mds_path, "rb"))
            else:
                Y["mds"] = binarize(mds(normed_tfidf, 70))
                pickle.dump(Y["mds"], open(self.config.mds_path, "wb"))

            pickle.dump(Y, open("../temp/sen_features.pkl", "wb"))
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
    tf.app.flags.DEFINE_string("embedding_file", "../data/word_embedding.txt", "embedding file txt")
    tf.app.flags.DEFINE_string("text_path", "../data/zhusu.txt", "file txt")
    tf.app.flags.DEFINE_string("ae_path", "../sen_ae.pkl", "pretrained average embedding features")
    tf.app.flags.DEFINE_string("le_path", "../sen_le.pkl", "pretrained laplacian eigenmaps features")
    tf.app.flags.DEFINE_string("lle_path", "../sen_lle.pkl", "pretrained Local Linear Embedding features")
    tf.app.flags.DEFINE_string("isomap_path", "../sen_isomap.pkl", "pretrained isomap features")
    tf.app.flags.DEFINE_string("mds_path", "../sen_mds.pkl", "pretrained multi dimemsional scaling features")
    tf.app.flags.DEFINE_string("lsa_path", "../sen_lsa.pkl", "pretrained latent semantic analysis features")
    tf.app.flags.DEFINE_integer("sample_num", 20000, "embedding words vocabulary")
    tf.app.flags.DEFINE_integer("voc_size", 200000, "embedding words vocabulary")
    tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimension")
    tf.app.flags.DEFINE_integer("feature_dim", 480, "cnn output features dimension")
    tf.app.flags.DEFINE_integer("target_dim", 70, "pretrained feature dimension")
    tf.app.flags.DEFINE_string("exp_name", "example", "exp name")
    tf.app.flags.DEFINE_integer("num_epochs", 500, "epochs")
    tf.app.flags.DEFINE_integer("num_iter_per_epoch", 100, "iter")
    tf.app.flags.DEFINE_float("learning_rate", 0.001, "lr")
    tf.app.flags.DEFINE_integer("batch_size", 200, "batch size")
    tf.app.flags.DEFINE_integer("max_to_keep", 5, "model to keep")
    tf.app.flags.DEFINE_integer("max_seq_len", 25, "max seq len")
    tf.app.flags.DEFINE_integer("cpu_num", 1, "cpu num")
    tf.app.flags.DEFINE_string("summary_dir", "../temp", "summary dir")
    tf.app.flags.DEFINE_string("checkpoint_dir", "../temp", "check point dir")
    tf.app.flags.DEFINE_integer("n_clusters", 20, "cluster nums")
    config = tf.app.flags.FLAGS
    generator = CharSenGenerator(config)
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
