# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: stc_model.py 
@time: 2018/12/17 
"""
import os
import pickle

import tensorflow as tf
from keras.layers import Conv1D, Dense, Layer, Flatten, InputSpec, AveragePooling1D
from keras.losses import binary_crossentropy
# from keras import layers
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from base.base_model import BaseModel


class StcModel(BaseModel):

    def __init__(self, config):
        super(StcModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        if os.path.exists("../temp/embedding.pkl"):
            self.embedding_matrix = pickle.load(open("../temp/embedding.pkl", "rb"))
        else:
            print("no embedding matrix")
            exit(-1)

        if os.path.exists("../temp/wi.pkl"):
            self.word_index = pickle.load(open("../temp/wi.pkl", "rb"))
        else:
            print("no word index dict")
            exit(-1)

        with tf.name_scope("inputs"):
            self.is_training = tf.placeholder(tf.bool)
            self.x = tf.placeholder(tf.int32, shape=[None, self.config.max_seq_len])
            self.y = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])

        with tf.name_scope("embedding_layer"):
            embeddings_var = tf.Variable(self.embedding_matrix, trainable=False, dtype=tf.float32)
            xx = tf.nn.embedding_lookup(embeddings_var, self.x)

        with tf.name_scope("cnn_layer"):
            x = Conv1D(100, 5, activation="tanh", padding="same")(xx)
            x = KMaxPooling(5)(x)
            # x._keras_shape = (None, 5, 100)  # fit some keras trick
            # x = Dropout(0.5)(x)
            x = Conv1D(100, 2, activation="tanh", padding="same")(x)
            x = tf.transpose(x, (0, 2, 1))
            x = AveragePooling1D(pool_size=2)(x)
            x = tf.transpose(x, (0, 2, 1))
            x = KMaxPooling(3)(x)
            x = Flatten()(x)
            # x = Dropout(0.5)(x)

        with tf.name_scope("dense_layer"):
            self.feature = Dense(self.config.feature_dim, activation="sigmoid")(x)
            self.pred = Dense(self.config.target_dim, activation="sigmoid")(x)

        with tf.name_scope("loss"):
            self.loss = binary_crossentropy(self.pred, self.y)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                             global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


class KMaxPooling(Layer):
    """
    k-max-pooling
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        top_k = tf.transpose(top_k, [0, 2, 1])
        return top_k

        # return flattened output
        # return Flatten()(top_k)


class KMeanModel(object):

    def __init__(self, config):
        self.config = config
        self.km = KMeans(n_clusters=config.n_clusters, n_jobs=config.cpu_num)

    def fit(self, features):
        V = normalize(features, norm="l2")
        self.km.fit(V)

    def predict(self, V):
        preds = self.km.predict(V)
        return preds
