import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        tf.app.flags.DEFINE_string("embedding_path",
                                   "/home/liuml/workspace/dialog_segment/sgns.merge.word/sgns.merge.word",
                                   "word embedding vector file path")
        tf.app.flags.DEFINE_integer("embedding_dim", 300, "word embedding dim")
        tf.app.flags.DEFINE_string("train_file_path", "/home/liuml/workspace/dialog_segment/data.csv",
                                   "train file path")
        tf.app.flags.DEFINE_integer("batch_size", 100, "mini batch size")
        tf.app.flags.DEFINE_integer("epoch_num", 3, "epoch num")
        tf.app.flags.DEFINE_integer("sen_size", 20, "sample sentence size")
        tf.app.flags.DEFINE_integer("sen_length", 20, "sentence length")
        tf.app.flags.DEFINE_integer("voc_size", 80000, "vocabulary dictionary size")
        tf.app.flags.DEFINE_integer("hidden_unit", 200, "lstm hidden unit")
        tf.app.flags.DEFINE_string("checkpoint_dir", "/home/liuml/workspace/dialog_segment", "model file path")
        tf.app.flags.DEFINE_string("model_name", "model.ckpt", "model file name")
        tf.app.flags.DEFINE_string("summary_dir", "/home/liuml/workspace/dialog_segment/log", "tensorboard log dir")
        tf.app.flags.DEFINE_boolean("train", "t", "train")
        tf.app.flags.DEFINE_integer("experiment_id", "1", "experiment id for 10 fold validation")

        config = tf.app.flags.FLAGS

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = ExampleModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
