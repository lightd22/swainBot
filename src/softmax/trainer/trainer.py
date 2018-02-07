import tensorflow as tf
import numpy as np
import random
import experience_replay as er
import match_processing as mp
import pandas as pd
import time

class Trainer():
    def __init__(self, sess, model, n_epoch, training_data, validation_data, batch_size):
        self._sess = sess
        self._model = model

        self._n_epoch = n_epoch
        self.epoch_counter = 0
        self._training_data = training_data
        self._validation_data = validation_data
        self._batch_size = batch_size
        self._buffer = er.ExperienceBuffer(max_buffer_size=20*len(training_data))
        self._val_buffer = er.ExperienceBuffer(max_buffer_size=20*len(validation_data))

        self.fill_buffer(training_data, self._buffer)
        self.fill_buffer(validation_data, self._val_buffer)

    def fill_buffer(self, data, buf):
        for match in data:
            for team in ["blue", "red"]:
                experiences = mp.process_match(match, team)
                # remove null actions (usually missing bans)
                for exp in experiences:
                    _,act,_,_ = exp
                    cid,pos = act
                    if(cid):
                        buf.store(exp)

    def sample_buffer(self, buf, n_samples):
        experiences = buf.sample(n_samples)
        states = []
        actions = []
        valid_actions = []
        for (state, action, _, _) in experiences:
            states.append(state.format())
            valid_actions.append(state.get_valid_actions())
            actions.append(state.get_action(*action))

        return (states, actions, valid_actions)

    def train(self):
        losses = []
        accs = {"train":[], "val":[]}
        for i in range(n_epoch):
            t0 =  time.time()
            loss, train_acc, val_acc = self.train_epoch()
            dt = time.time()-t0
            losses.append(loss)
            accs["train"].append(train_acc)
            accs["val"].append(val_acc)
            print(" Finished epoch {}/{}: dt {:.2f}, loss {:.6f}, train {:.6f}, val {:.6f}".format(i+1, self._n_epoch, dt, loss, train_acc, val_acc), flush=True)

        return (losses, accs)

    def train_epoch(self):
        n_iter = self._buffer.buffer_size // self._batch_size

        for it in range(n_iter):
            self.train_step()

        self.epoch_counter += 1
        loss, train_acc = self.validate(self._buffer)
        _, val_acc = self.validate(self._val_buffer)

        return (loss, train_acc, val_acc)

    def validate(self, buf):
        states, actions, valid_actions = self.sample_buffer(buf, buf.get_buffer_size())

        feed_dict = {self._model.input:np.stack(states, axis=0),
                     self._model.valid_actions:np.stack(valid_actions, axis=0),
                     self._model.actions:actions}
        loss, train_probs = self._sess.run([self._model.loss, self._model.probabilities], feed_dict=feed_dict)

        THRESHOLD = 5
        accurate_predictions = 0
        for k in range(len(states)):
            probabilities = train_probs[k,:]
            data = [(a, probabilities[a]) for a in range(len(probabilities))]
            df = pd.DataFrame(data, columns=['act_id','prob'])

            df.sort_values('prob',ascending=False,inplace=True)
            df.reset_index(drop=True,inplace=True)
            df['rank'] = df.index

            submitted_action_id = actions[k]
            submitted_row = df[df['act_id']==submitted_action_id]

            rank = submitted_row['rank'].iloc[0]
            if(rank < THRESHOLD):
                accurate_predictions += 1

        accuracy = accurate_predictions/len(states)
        return (loss, accuracy)

    def train_step(self):
        states, actions, valid_actions = self.sample_buffer(self._buffer, self._batch_size)

        feed_dict = {self._model.input:np.stack(states, axis=0),
                     self._model.valid_actions:np.stack(valid_actions, axis=0),
                     self._model.actions:actions,
                     self._model.dropout_keep_prob:0.5}
         _ = self._sess.run(self._model.update, feed_dict=feed_dict)
