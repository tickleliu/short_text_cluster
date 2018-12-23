import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from base.base_train import BaseTrain


class Stc2Trainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(Stc2Trainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
            # accs.append(acc)
        loss = np.mean(losses)
        # acc = np.mean(accs)
        print("loss: %s" % loss)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y_ae, batch_y_lle, batch_y_le, batch_y_isomap, batch_y_lsa, batch_y_mds = next(
            self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x,
                     self.model.y_ae: batch_y_ae,
                     self.model.y_lle: batch_y_lle,
                     self.model.y_le: batch_y_le,
                     self.model.y_isomap: batch_y_isomap,
                     self.model.y_lsa: batch_y_lsa,
                     self.model.y_mds: batch_y_mds,
                     self.model.is_training: True}
        # _, loss = self.sess.run([self.model.train_step, self.model.loss],
        #                         feed_dict=feed_dict, options=self.options, run_metadata=self.run_metadata)
        _, loss, grad_a = self.sess.run([self.model.train_step, self.model.loss, self.model.cnn1_grads_a],
                                        feed_dict=feed_dict)
        # print(loss)
        # print(grad_a)
        return loss

    def inference(self, sen: str):
        x = sen
        # sen = sen.split(" ")
        # x = []
        # for word in sen:
        #     if word in self.model.word_index:
        #         x.append(self.model.word_index[word])
        #     else:
        #         x.append(0)
        # x = np.asarray(x)
        x = np.expand_dims(x, 0)
        # x = pad_sequences(x, maxlen=self.config.max_seq_len)

        feed_dict = {self.model.x: x, self.model.is_training: False}
        result = self.sess.run([self.model.feature],
                               feed_dict=feed_dict)
        return result
