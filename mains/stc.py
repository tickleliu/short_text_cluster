# -*- coding:utf-8 -*-
"""
@author:mlliu
@file: stc.py
@time: 2018/11/30
"""
import tensorflow as tf

from tqdm import tqdm
from data_loader.char_sen_generator import CharSenGenerator
from models.stc_model import StcModel, KMeanModel
import json
from trainers.stc_trainer import StcTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import numpy as np


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        tf.app.flags.DEFINE_string("embedding_file", "../data/word_embedding.txt", "embedding file txt")
        tf.app.flags.DEFINE_string("text_path", "../data/sens.npy", "embedding file txt")
        tf.app.flags.DEFINE_integer("max_seq_len", 30, "seq length")
        tf.app.flags.DEFINE_integer("voc_size", 2000, "embedding words vocabulary")
        tf.app.flags.DEFINE_integer("emb_dim", 300, "word embedding dimension")
        tf.app.flags.DEFINE_integer("target_dim", 300, "word embedding dimension")
        tf.app.flags.DEFINE_string("exp_name", "example", "exp name")
        tf.app.flags.DEFINE_integer("num_epochs", 1, "epochs")
        tf.app.flags.DEFINE_integer("num_iter_per_epoch", 200, "iter")
        tf.app.flags.DEFINE_float("learning_rate", 0.001, "lr")
        tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
        tf.app.flags.DEFINE_integer("max_to_keep", 5, "model to keep")
        tf.app.flags.DEFINE_string("summary_dir", "../temp", "summary dir")
        tf.app.flags.DEFINE_string("checkpoint_dir", "../temp", "check point dir")
        tf.app.flags.DEFINE_integer("n_clusters", 1000, "cluster nums")
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
    sess = tf.Session()
    # create your data generator
    data = CharSenGenerator(config)
    # create an instance of the model you want
    model = StcModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = StcTrainer(sess, model, data, config, logger)
    # load model if exists
    model.load(sess)
    # here you train your model
    # trainer.train()

    size, len = data.input.shape
    result = []
    for i in tqdm(range(size), total=size):
        sen = data.data[i]
        p = trainer.inference(sen)
        p = p[0]
        result.append(p)

    # V = np.asarray(result, dtype=np.float32)
    V = np.concatenate(tuple(result))
    km = KMeanModel(config)
    print("cluster %s text to %s clusters" % (size, config.n_clusters))
    km.fit(V)
    print("predict text into clusters")
    pred = km.predict(V)
    with open("result.json", "w") as f:
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
