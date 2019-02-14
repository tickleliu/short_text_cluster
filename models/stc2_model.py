# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: stc_model.py 
@time: 2018/12/17 
"""
import os
import pickle

import tensorflow as tf
from keras.layers import Conv1D, Dense, Layer, Flatten, Activation, InputSpec, GlobalMaxPooling1D, AveragePooling1D, \
    Dropout, \
    BatchNormalization, Embedding
from keras.losses import binary_crossentropy
# from keras import layers
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from base.base_model import BaseModel
import numpy as np


class Stc2Model(BaseModel):
    def __init__(self, config):
        super(Stc2Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        if os.path.exists("../temp/sen_embedding.pkl"):
            self.embedding_matrix = pickle.load(open("../temp/sen_embedding.pkl", "rb"))
        else:
            print("no embedding matrix")
            exit(-1)

        if os.path.exists("../temp/sen_wi.pkl"):
            self.word_index = pickle.load(open("../temp/sen_wi.pkl", "rb"))
        else:
            print("no word index dict")
            exit(-1)

        with tf.name_scope("inputs"):
            self.is_training = tf.placeholder(tf.bool)
            self.x = tf.placeholder(tf.int32, shape=[None, self.config.max_seq_len])
            self.y_ae = tf.placeholder(tf.float32, shape=[None, self.config.embedding_dim])
            self.y_le = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])
            self.y_lle = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])
            self.y_isomap = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])
            self.y_mds = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])
            self.y_lsa = tf.placeholder(tf.float32, shape=[None, self.config.target_dim])

        with tf.name_scope("embedding_layer"):
            embeddings_var = tf.Variable(self.embedding_matrix, trainable=False, dtype=tf.float32)
            xx = tf.nn.embedding_lookup(embeddings_var, self.x)
            # xx = Embedding(input_dim=11366, output_dim=300, weights=[self.embedding_matrix],
            #                input_length=self.config.max_seq_len, trainable=False)(self.x)

        with tf.name_scope("cnn_layer"):
            x = Conv1D(100, 5, activation="tanh", padding="same")(xx)
            x = KMaxPooling(5)(x)
            x = Dropout(0.5)(x)
            x = Conv1D(100, 2, activation="tanh", padding="same")(x)
            x = tf.transpose(x, (0, 2, 1))
            x = AveragePooling1D(pool_size=2)(x)
            x = tf.transpose(x, (0, 2, 1))
            x = KMaxPooling(3)(x)
            x = Flatten()(x)
            x = Dropout(0.5)(x)
            self.feature = x

            # x = Conv1D(100, 5, activation='tanh', padding='same')(xx)
            # x = Dropout(0.5)(x)
            # x = Conv1D(100, 5, activation='tanh', padding='same')(x)
            # # Output
            # x = Dropout(0.5)(x)
            # x = GlobalMaxPooling1D()(x)
            # x = Dense(480, activation="sigmoid")(x)
            # self.feature = x
        with tf.name_scope("dense_layer"):
            # self.feature = Dense(self.config.feature_dim, activation="sigmoid")(x)
            self.pred_ae = Dense(self.config.embedding_dim, activation="sigmoid")(x)
            self.pred_le = Dense(self.config.target_dim, activation="sigmoid")(x)
            self.pred_lle = Dense(self.config.target_dim, activation="sigmoid")(x)
            self.pred_isomap = Dense(self.config.target_dim, activation="sigmoid")(x)
            self.pred_lsa = Dense(self.config.target_dim, activation="sigmoid")(x)
            self.pred_mds = Dense(self.config.target_dim, activation="sigmoid")(x)
            # self.feature = self.pred_ae

        with tf.name_scope("loss"):
            # transform back to logits
            loss_ae = self.binary_crossentropy(self.pred_ae, self.y_ae)
            loss_le = self.binary_crossentropy(self.pred_le, self.y_le)
            loss_lle = self.binary_crossentropy(self.pred_lle, self.y_lle)
            loss_isomap = self.binary_crossentropy(self.pred_isomap, self.y_isomap)
            loss_mds = self.binary_crossentropy(self.pred_mds, self.y_mds)
            loss_lsa = self.binary_crossentropy(self.pred_lsa, self.y_lsa)
            self.loss = (loss_ae + loss_le + loss_lle + loss_isomap + loss_mds + loss_lsa) / 6
            # self.loss = loss_ae

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(self.config.learning_rate, beta1=0.9, beta2=0.999,
                                                   epsilon=1e-08)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                cnn1_grads = grads_and_vars[0][0]
                cnn1_grads_vector = Flatten()(cnn1_grads)
                cnn1_grads_square = tf.multiply(cnn1_grads_vector, cnn1_grads_vector)
                self.cnn1_grads_a = tf.sqrt(tf.reduce_mean(cnn1_grads_square))
                # grads_and_vars = tf.Print(grads_and_vars, [cnn1_grads_a], "cnn1 grad")

                self.train_step = optimizer.apply_gradients(grads_and_vars,
                                                            global_step=self.global_step_tensor)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def binary_crossentropy(self, output, target):
        from keras.backend.tensorflow_backend import _to_tensor
        from keras.backend.common import epsilon
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                                      logits=output))
        return loss


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
        return self.km.labels_
