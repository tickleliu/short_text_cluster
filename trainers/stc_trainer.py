from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class StcTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(StcTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
            # accs.append(acc)
        loss = np.mean(losses)
        # acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                feed_dict=feed_dict)
        return loss

    def inference(self, sen: str):
        if type(sen) is list:
            chars = sen
        elif type(sen) is str:
            sen = sen.replace(" ", "")
            chars = [char for char in sen]
        x = []
        for char in chars:
            if char in self.model.word_index:
                x.append(self.model.word_index[char])
            else:
                x.append(0)
        x = np.asarray(x)
        x = np.expand_dims(x, 0)
        x = pad_sequences(x, maxlen=self.config.max_seq_len)

        feed_dict = {self.model.x: x, self.model.is_training: False}
        result = self.sess.run([self.model.pred],
                               feed_dict=feed_dict)
        return result
