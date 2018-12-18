# -*- coding:utf-8 -*-
""" 
@author:mlliu
@file: stc_model.py 
@time: 2018/12/17 
"""

import tensorflow as tf
from base.base_model import BaseModel
from keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D
from keras.losses import binary_crossentropy


class STC2Model(BaseModel):

    def __init__(self, config):
        super(STC2Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.name_scope("inputs"):
            self.is_training = tf.placeholder(tf.bool)
            self.x = tf.placeholder(tf.float32, shape=[None, self.config.seq_len])
            self.y = tf.placeholder(tf.float32, shape=[None, self.config.reduce_dim])
        with tf.name_scope("embedding_layer"):
            embeddings_var = tf.get_variable(name="embedding_variables",
                                             initializer=tf.truncated_normal(
                                                 shape=[self.config.voc_size, self.config.emb_dim], stddev=0.1),
                                             trainable=True)
            x = tf.nn.embedding_lookup(embeddings_var, self.x)
        with tf.name_scope("cnn_layer"):
            x = Conv1D(filters=[100], kernel_size=(3), activation="relu", padding="same")(x)
            x = GlobalAveragePooling1D()(x)
        with tf.name_scope("dense_layer"):
            self.pred = Dense()(x)

        with tf.name_scope("loss"):
            # self.loss = loss + l2_loss
            # loss_r2 =
            update_ops = tf.get_collectiont(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                pass

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
