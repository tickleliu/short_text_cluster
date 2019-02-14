# -*- coding:utf-8 -*-
"""
@author:mlliu
@file: stc.py
@time: 2018/11/30
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_loader.char_sen_generator import CharSenGenerator
from models.stc2_model import Stc2Model, KMeanModel
from trainers.stc2_trainer import Stc2Trainer
from utils.dirs import create_dirs
from utils.logger import Logger
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize
from utils.utils import cluster_quality


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        tf.app.flags.DEFINE_string("embedding_file", "../data/word_embedding.txt", "embedding file txt")
        tf.app.flags.DEFINE_string("text_path", "../data/zhusu.txt", "file txt")
        tf.app.flags.DEFINE_string("ae_path", "../temp/sen_ae.pkl", "pretrained average embedding features")
        tf.app.flags.DEFINE_string("le_path", "../temp/sen_le.pkl", "pretrained laplacian eigenmaps features")
        tf.app.flags.DEFINE_string("lle_path", "../temp/sen_lle.pkl", "pretrained Local Linear Embedding features")
        tf.app.flags.DEFINE_string("isomap_path", "../temp/sen_isomap.pkl", "pretrained isomap features")
        tf.app.flags.DEFINE_string("mds_path", "../temp/sen_mds.pkl", "pretrained multi dimemsional scaling features")
        tf.app.flags.DEFINE_string("lsa_path", "../temp/sen_lsa.pkl", "pretrained latent semantic analysis features")
        tf.app.flags.DEFINE_integer("sample_num", 20000, "embedding words vocabulary")
        tf.app.flags.DEFINE_integer("voc_size", 200000, "embedding words vocabulary")
        tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dimension")
        tf.app.flags.DEFINE_integer("feature_dim", 480, "cnn output features dimension")
        tf.app.flags.DEFINE_integer("target_dim", 70, "pretrained feature dimension")
        tf.app.flags.DEFINE_string("exp_name", "example", "exp name")
        tf.app.flags.DEFINE_integer("num_epochs", 50, "epochs")
        tf.app.flags.DEFINE_integer("num_iter_per_epoch", 100, "iter")
        tf.app.flags.DEFINE_float("learning_rate", 0.001, "lr")
        tf.app.flags.DEFINE_integer("batch_size", 200, "batch size")
        tf.app.flags.DEFINE_integer("max_to_keep", 5, "model to keep")
        tf.app.flags.DEFINE_integer("max_seq_len", 25, "max seq len")
        tf.app.flags.DEFINE_integer("cpu_num", 4, "cpu num")
        tf.app.flags.DEFINE_string("summary_dir", "../temp", "summary dir")
        tf.app.flags.DEFINE_string("checkpoint_dir", "../temp", "check point dir")
        tf.app.flags.DEFINE_integer("n_clusters", 20000, "cluster nums")
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
    data = CharSenGenerator(config)
    # create an instance of the model you want
    model = Stc2Model(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = Stc2Trainer(sess, model, data, config, logger)
    # here you train your model
    size = len(data.data)

    word_index = pickle.load(open("../temp/sen_wi.pkl", "rb"))  # type:dict
    use_mode = "model"
    if use_mode == "model":

        # load model if exists
        model.load(sess)
        trainer.train()
        result = []
        result_dict = {}
        for i in tqdm(range(size), total=size):
            class_sen = data.data[i:i + 1]  # type:str
            # sen = data.input[i]
            sequences_full = data.tokenizer.texts_to_sequences(class_sen)
            sen = pad_sequences(sequences_full, maxlen=config.max_seq_len)
            p = trainer.inference(sen)
            p = p[0]
            result.append(p)
            class_sen = class_sen[0].replace(" ", "")
            result_dict[class_sen] = p
        pickle.dump(result_dict, open("vector.pkl", "wb"))
        V = np.concatenate(tuple(result))
    else:
        V = data.y[use_mode]
    # V = normalize(V, norm='l2')
    km = KMeanModel(config)
    print("cluster %s text to %s clusters" % (size, config.n_clusters))
    print(np.shape(V))
    km.fit(V[0:100000])
    print("predict text into clusters")
    pred = km.predict(V)
    # a = {'deep': cluster_quality(data.target, pred)}
    index_word = dict(zip(word_index.items(), word_index.keys()))
    with open("%s.json" % use_mode, "w") as f:
        class_dict = {}
        for index, class_num in enumerate(pred):
            class_sens = class_dict.get(class_num, [])
            class_sen = data.data[index]  # type:str
            class_sen = class_sen.replace(" ", "")
            class_sens.append(class_sen)
            class_dict[class_num] = class_sens
        for key, value in class_dict.items():
            f.writelines(json.dumps({str(key): value}, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
