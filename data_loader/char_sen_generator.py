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


class CharSenGenerator(DataGenerator):

    def __init__(self, config):
        super(CharSenGenerator, self).__init__(config)
        self.config = config
        self.embedding_file = config.embedding_file
        self.text_path = config.text_path

        with open(self.text_path, "rb") as f:
            sens = pickle.load(f)
        data = []
        for sen in sens:
            line = []
            for word in sen:
                line.extend([char for char in word])
            line = " ".join(line)
            line = line[0:config.max_seq_len]
            data.append(line)

        print("total: %s short texts" % format(len(data), ","))
        self.data = data

        tokenizer = Tokenizer(char_level=False)
        tokenizer.fit_on_texts(data)
        sequences_full = tokenizer.texts_to_sequences(data)
        word_index = tokenizer.word_index
        pickle.dump(word_index, open("../temp/wi.pkl", "wb"))
        print('Found %s unique tokens.' % len(word_index))
        MAX_NB_WORDS = len(word_index)
        self.input = pad_sequences(sequences_full, maxlen=config.max_seq_len)

        print('Preparing embedding matrix')
        if os.path.exists("../temp/embedding.pkl"):
            embedding_matrix = pickle.load(open("../temp/embedding.pkl", "rb"))
        else:
            nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
            embedding_matrix = np.zeros((nb_words, config.emb_dim))
            word2vec = {}

            topn = config.voc_size
            lines_num = 0
            with open(self.embedding_file, encoding='utf-8', errors='ignore') as f:
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
            pickle.dump(embedding_matrix, open("../temp/embedding.pkl", "wb"))
            print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

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

            # normed_tfidf = pca(normed_tfidf, 800)
            # print("calc lle")
            # if os.path.exists("../temp/sen_lle.pkl"):
            #     Y["lle"] = pickle.load(open("../temp/sen_lle.pkl", "rb"))
            # else:
            #     Y["lle"] = binarize(lle(normed_tfidf, 128))
            #     pickle.dump(Y["lle"], open("../temp/sen_lle.pkl", "wb"))
            #
            # print("calc isomap")
            # if os.path.exists("../temp/sen_isomap.pkl"):
            #     Y["isomap"] = pickle.load(open("../temp/sen_isomap.pkl", "rb"))
            # else:
            #     Y["isomap"] = binarize(isomap(normed_tfidf, 128))
            #     pickle.dump(Y["isomap"], open("../temp/sen_isomap.pkl", "wb"))
            #
            # print("calc mds")
            # if os.path.exists("../temp/sen_mds.pkl"):
            #     Y["mds"] = pickle.load(open("../temp/sen_mds.pkl", "rb"))
            # else:
            #     Y["mds"] = binarize(mds(normed_tfidf, 128))
            #     pickle.dump(Y["mds"], open("../temp/sen_mds.pkl", "wb"))
            #
            # print("calc le")
            # if os.path.exists("../temp/sen_le.pkl"):
            #     Y["le"] = pickle.load(open("../temp/sen_le.pkl", "rb"))
            # else:
            #     Y["le"] = binarize(le(normed_tfidf, 128))
            #     pickle.dump(Y["le"], open("../temp/sen_le.pkl", "wb"))
            #
            # print("calc tsne")
            # if os.path.exists("../temp/sen_tsne.pkl"):
            #     Y["tsne"] = pickle.load(open("../temp/sen_tsne.pkl", "rb"))
            # else:
            #     Y["tsne"] = binarize(tsne(normed_tfidf, 128))
            #     pickle.dump(Y["tsne"], open("../temp/sen_tsne.pkl", "wb"))

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
                yield self.input[idx], self.y[self.config.reduce_func][idx]
                i += batch_size
            else:
                i = 0
                indices = np.arange(size)
                np.random.shuffle(indices)


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("embedding_file", "../data/word_embedding.txt", "embedding file txt")
    tf.app.flags.DEFINE_string("text_path", "../data/sens.npy", "embedding file txt")
    tf.app.flags.DEFINE_integer("max_seq_len", 30, "seq length")
    tf.app.flags.DEFINE_integer("embedding_voc", 2000, "embedding words vocabulary")
    tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimension")
    config = tf.app.flags.FLAGS
    generator = CharSenGenerator(config)
    # generator_next_batch = generator.next_batch(100)
    # while True:
    #     x, y = next(generator_next_batch)
    #     print(np.shape(x))
    #     print(np.shape(y))
    #     print(y)
