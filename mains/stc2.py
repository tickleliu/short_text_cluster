# -*- coding:utf-8 -*-
"""
@author:mlliu
@file: stc.py
@time: 2018/11/30
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader.stackoverflow_generator import StackGenerator
from models.stc2_model import Stc2Model, KMeanModel
from trainers.stc2_trainer import Stc2Trainer
from utils.dirs import create_dirs
from utils.logger import Logger
from sklearn.preprocessing import normalize
from utils.utils import cluster_quality


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        tf.app.flags.DEFINE_string("embedding_file", "../data/GoogleNews-vectors-negative300.bin", "embedding file txt")
        tf.app.flags.DEFINE_string("text_path", "../data/StackOverflow.txt", "embedding file txt")
        tf.app.flags.DEFINE_string("label_path", "../data/StackOverflow_gnd.txt", "embedding file txt")
        tf.app.flags.DEFINE_string("ae_path", "../ae.pkl", "pretrained average embedding features")
        tf.app.flags.DEFINE_string("le_path", "../le.pkl", "pretrained laplacian eigenmaps features")
        tf.app.flags.DEFINE_string("lle_path", "../lle.pkl", "pretrained Local Linear Embedding features")
        tf.app.flags.DEFINE_string("isomap_path", "../isomap.pkl", "pretrained isomap features")
        tf.app.flags.DEFINE_string("mds_path", "../mds.pkl", "pretrained multi dimemsional scaling features")
        tf.app.flags.DEFINE_string("lsa_path", "../lsa.pkl", "pretrained latent semantic analysis features")
        tf.app.flags.DEFINE_integer("voc_size", 2000, "embedding words vocabulary")
        tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimension")
        tf.app.flags.DEFINE_integer("feature_dim", 480, "cnn output features dimension")
        tf.app.flags.DEFINE_integer("target_dim", 70, "pretrained feature dimension")
        tf.app.flags.DEFINE_string("exp_name", "example", "exp name")
        tf.app.flags.DEFINE_integer("num_epochs", 500, "epochs")
        tf.app.flags.DEFINE_integer("num_iter_per_epoch", 100, "iter")
        tf.app.flags.DEFINE_float("learning_rate", 0.001, "lr")
        tf.app.flags.DEFINE_integer("batch_size", 200, "batch size")
        tf.app.flags.DEFINE_integer("max_to_keep", 5, "model to keep")
        tf.app.flags.DEFINE_integer("cpu_num", 1, "cpu num")
        tf.app.flags.DEFINE_string("summary_dir", "../temp", "summary dir")
        tf.app.flags.DEFINE_string("checkpoint_dir", "../temp", "check point dir")
        tf.app.flags.DEFINE_integer("n_clusters", 20, "cluster nums")
        tf.app.flags.DEFINE_string("reduce_func", "ae", "reduce function")
        config = tf.app.flags.FLAGS
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    # create your data generator
    data = StackGenerator(config)
    # create an instance of the model you want
    model = Stc2Model(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Stc2Trainer(sess, model, data, config, logger)
    # load model if exists
    # model.load(sess)
    # here you train your model
    trainer.train()
    # from keras.layers import Input, Embedding, Flatten, Reshape
    # from keras.layers import Dense, Conv1D, Dropout, merge
    # from keras.layers import MaxPooling1D, GlobalMaxPooling1D
    # from keras.models import Model
    # from keras.optimizers import Adam
    #
    # trainable_embedding = False
    # # Embedding ler
    # pretrained_embedding_layer = Embedding(
    #     input_dim=11366,
    #     output_dim=300,
    #     weights=[data.embedding_matrix],
    #     input_length=config.max_seq_len,
    # )
    #
    # # Input
    # sequence_input = Input(shape=(config.max_seq_len,), dtype='int32')
    # embedded_sequences = pretrained_embedding_layer(sequence_input)
    # x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    # x = Dropout(0.5)(x)
    # x = Conv1D(100, 5, activation='tanh', padding='same')(x)
    # # Output
    # x = Dropout(0.5)(x)
    # x = GlobalMaxPooling1D()(x)
    # deepfeatures = Dense(480, activation="sigmoid")(x)
    #
    # # Output
    # # x = Dropout(0.5)(x)
    # predictions = Dense(300, activation='sigmoid')(deepfeatures)
    # model = Model(sequence_input, predictions)
    #
    # model.layers[1].trainable = trainable_embedding
    #
    # adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # # Loss and Optimizer
    # model.compile(loss='binary_crossentropy',
    #               optimizer=adam,
    #               metrics=['mae'])
    # model.fit(data.input, data.y["ae"], validation_split=0.2,
    #           epochs=1, batch_size=200, verbose=1, shuffle=True)
    # input = model.layers[0].input
    # output = model.layers[-2].output
    # model_penultimate = Model(input, output)
    # V = model_penultimate.predict(data.input)

    size, _ = data.input.shape

    use_mode = "model"
    if use_mode == "model":
        result = []
        for i in tqdm(range(size), total=size):
            sen = data.input[i]
            p = trainer.inference(sen)
            p = p[0]
            result.append(p)
        V = np.concatenate(tuple(result))
    else:
        V = data.y[use_mode]
    V = normalize(V, norm='l2')
    km = KMeanModel(config)
    print("cluster %s text to %s clusters" % (size, config.n_clusters))
    print(np.shape(V))
    km.fit(V)
    print("predict text into clusters")
    pred = km.predict(V)
    a = {'deep': cluster_quality(data.target, pred)}
    # with open("%s.json" % use_mode, "w") as f:
    #     class_dict = {}
    #     for index, class_num in enumerate(pred):
    #         class_sens = class_dict.get(class_num, [])
    #         class_sen = data.data[index]  # type:str
    #         class_sen = class_sen.replace(" ", "")
    #         class_sens.append(class_sen)
    #         class_dict[class_num] = class_sens
    #     for key, value in class_dict.items():
    #         f.writelines(json.dumps({str(key): value}, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
