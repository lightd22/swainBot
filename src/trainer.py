import time
import random
from copy import deepcopy

import tensorflow as tf
import pandas as pd
import numpy as np

import data.match_pool as pool

from features.draftstate import DraftState
import features.experience_replay as er
import features.match_processing as mp
from features.rewards import get_reward

class BaseTrainer():
    pass

class DDQNTrainer(BaseTrainer):
    """
    Trainer class for Double DQN networks.
    Args:
        q_network (qNetwork): Q-network containing "online" and "target" networks
        n_epochs (int): number of times to iterate through given data
        training_matches (list(match)): list of matches to be trained on
        validation_matches (list(match)): list of matches to validate model against
        batch_size (int): size of each training set sampled from the replay buffer which will be used to update Qnet at a time
        buffer_size (int): size of replay buffer used
        load_path (string): path to reload existing model
    """
    def __init__(self, q_network, n_epoch, training_data, validation_data, batch_size, buffer_size, load_path=None):
        num_episodes = len(training_data)
        print("***")
        print("Beginning training..")
        print("  train_epochs: {}".format(n_epoch))
        print("  num_episodes: {}".format(num_episodes))
        print("  batch_size: {}".format(batch_size))
        print("  buffer_size: {}".format(buffer_size))
        print("***")

        self.ddq_net = q_network
        self.n_epoch = n_epoch
        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.load_path = load_path

        self.replay = er.ExperienceBuffer(self.buffer_size)
        self.step_count = 0
        self.epoch_count = 0

        self.dampen_states = False
        self.teams = [DraftState.BLUE_TEAM, DraftState.RED_TEAM]

        self.N_TEMP_TRAIN_MATCHES = 25
        self.TEMP_TRAIN_PATCHES = ["8.13","8.14","8.15"]

    def train(self):
        """
        Core training loop over epochs
        """
        self.target_update_frequency = 10000 # How often to update target network

        stash_model = True # Flag for stashing a copy of the model
        model_stash_interval = 10 # Stashes a copy of the model this often

        # Number of steps to take before training. Allows buffer to partially fill.
        # Must be at least batch_size to avoid error when sampling from experience replay
        self.pre_training_steps = 10*self.batch_size
        assert(self.pre_training_steps <= self.buffer_size), "Replay not large enough for pre-training!"
        assert(self.pre_training_steps >= self.batch_size), "Buffer not allowed to fill enough before sampling!"
        # Number of steps to force learner to observe submitted actions, rather than submit its own actions
        self.observations = 2000

        self.epsilon = 0.5 # Initial probability of letting the learner submit its own action
        self.eps_decay_rate = 1./(25*20*len(self.training_data)) # Rate at which epsilon decays per submission

        lr_decay_freq = 10 # Decay learning rate after a set number of epochs
        min_learning_rate = 1.e-8 # Minimum learning rate allowed to decay to

        summaries = {}
        summaries["loss"] = []
        summaries["train_acc"] = []
        summaries["val_acc"] = []
        # Load existing model
        self.ddq_net.sess.run(self.ddq_net.online_ops["init"])
        if(self.load_path):
            self.ddq_net.load(self.load_path)
            print("\nCheckpoint loaded from {}".format(self.load_path))

        # Initialize target network
        self.ddq_net.sess.run(self.ddq_net.target_ops["target_init"])

        for self.epoch_count in range(self.n_epoch):
            t0 = time.time()
            learning_rate = self.ddq_net.online_ops["learning_rate"].eval(self.ddq_net.sess)
            if((self.epoch_count>0) and (self.epoch_count % lr_decay_freq == 0) and (learning_rate>= min_learning_rate)):
                # Decay learning rate accoring to schedule
                learning_rate = 0.5*learning_rate
                self.ddq_net.sess.run(self.ddq_net.online_ops["learning_rate"].assign(learning_rate))

            # Run single epoch of training
            loss, train_acc, val_acc = self.train_epoch()
            dt = time.time()-t0

            print(" Finished epoch {:2}/{}: lr: {:.4e}, dt {:.2f}, loss {:.6f}, train {:.6f}, val {:.6f}".format(self.epoch_count+1, self.n_epoch, learning_rate, dt, loss, train_acc, val_acc), flush=True)
            summaries["loss"].append(loss)
            summaries["train_acc"].append(train_acc)
            summaries["val_acc"].append(val_acc)

            if(stash_model):
                if(self.epoch_count>0 and (self.epoch_count+1)%model_stash_interval==0):
                    # Stash a copy of the current model
                    out_path = "tmp/models/{}_model_E{}.ckpt".format(self.ddq_net._name, self.epoch_count+1)
                    self.ddq_net.save(path=out_path)
                    print("Stashed a copy of the current model in {}".format(out_path))

        self.ddq_net.save(path=self.ddq_net._path_to_model)
        return summaries

    def train_epoch(self):
        """
        Training loop for a single epoch
        """
        # We can't validate a winner for submissions generated by the learner,
        # so we will use a winner-less match when getting rewards for such states
        blank_match = {"winner":None}

        learner_submitted_actions = 0
        null_actions = 0

        # Shuffle match presentation order
        if(self.N_TEMP_TRAIN_MATCHES):
            path_to_db = "../data/competitiveMatchData.db"
            sources = {"patches":self.TEMP_TRAIN_PATCHES, "tournaments":[]}
            print("Adding {} matches to training pool from {}.".format(self.N_TEMP_TRAIN_MATCHES, path_to_db))
            temp_matches = pool.match_pool(self.N_TEMP_TRAIN_MATCHES, path_to_db, randomize=True, match_sources=sources)["matches"]
        else:
            temp_matches = []
        data = self.training_data + temp_matches

        shuffled_matches = random.sample(data, len(data))
        for match in shuffled_matches:
            for team in self.teams:
                # Process match into individual experiences
                experiences = mp.process_match(match, team)
                for pick_id, experience in enumerate(experiences):
                    # Some experiences include NULL submissions (usually missing bans)
                    # The learner isn't allowed to submit NULL picks so skip adding these
                    # to the buffer.
                    state,actual,_,_ = experience
                    (cid,pos) = actual
                    if cid is None:
                        null_actions += 1
                        continue
                    # Store original experience
                    self.replay.store([experience])
                    self.step_count += 1

                    # Give model feedback on current estimations
                    if(self.step_count > self.observations):
                        # Let the network predict the next action
                        feed_dict = {self.ddq_net.online_ops["input"]:[state.format_state()],
                                     self.ddq_net.online_ops["valid_actions"]:[state.get_valid_actions()]}
                        q_vals = self.ddq_net.sess.run(self.ddq_net.online_ops["valid_outQ"], feed_dict=feed_dict)
                        sorted_actions = q_vals[0,:].argsort()[::-1]
                        top_actions = sorted_actions[0:4]

                        if(random.random() < self.epsilon):
                            pred_act = random.sample(list(top_actions), 1)
                        else:
                            # Use model's top prediction
                            pred_act = [sorted_actions[0]]

                        for action in pred_act:
                            (cid,pos) = state.format_action(action)
                            if((cid,pos)!=actual):
                                pred_state = deepcopy(state)
                                pred_state.update(cid,pos)
                                r = get_reward(pred_state, blank_match, (cid,pos), actual)
                                new_experience = (state, (cid,pos), r, pred_state)

                                self.replay.store([new_experience])
                                learner_submitted_actions += 1

                    if(self.epsilon > 0.1):
                        # Reduce epsilon over time
                        self.epsilon -= self.eps_decay_rate

                    # Use minibatch sample to update online network
                    if(self.step_count > self.pre_training_steps):
                        self.train_step()

                    if(self.step_count % self.target_update_frequency == 0):
                        # After the online network has been updated, update target network
                        _ = self.ddq_net.sess.run(self.ddq_net.target_ops["target_update"])

        # Get training loss, training_acc, and val_acc to return
        loss, train_acc = self.validate_model(self.training_data)
        _, val_acc = self.validate_model(self.validation_data)
        return (loss, train_acc, val_acc)

    def train_step(self):
        """
        Training logic for a single mini-batch update sampled from replay
        """
        # Sample training batch from replay
        training_batch = self.replay.sample(self.batch_size)

        # Calculate target Q values for each example:
        # For non-terminal states, targetQ is estimated according to
        #   targetQ = r + gamma*Q'(s',max_a Q(s',a))
        # where Q' denotes the target network.
        # For terminating states the target is computed as
        #   targetQ = r
        updates = []
        for exp in training_batch:
            start,_,reward,end = exp
            if(self.dampen_states):
                # To dampen states (usually done after major patches or when the meta shifts)
                # we replace winning rewards with 0.
                reward = 0.
            state_code = end.evaluate()
            if(state_code==DraftState.DRAFT_COMPLETE or state_code in DraftState.invalid_states):
                # Action moves to terminal state
                updates.append(reward)
            else:
                # Follwing double DQN paper (https://arxiv.org/abs/1509.06461).
                #  Action is chosen by online network, but the target network is used to evaluate this policy.
                # Each row in predicted_Q gives estimated Q(s',a) values for all possible actions for the input state s'.
                feed_dict = {self.ddq_net.online_ops["input"]:[end.format_state()],
                             self.ddq_net.online_ops["valid_actions"]:[end.get_valid_actions()]}
                predicted_action = self.ddq_net.sess.run(self.ddq_net.online_ops["prediction"], feed_dict=feed_dict)[0]

                feed_dict = {self.ddq_net.target_ops["input"]:[end.format_state()]}
                predicted_Q = self.ddq_net.sess.run(self.ddq_net.target_ops["outQ"], feed_dict=feed_dict)

                updates.append(reward + self.ddq_net.discount_factor*predicted_Q[0,predicted_action])

        # Update online net using target Q
        # Experience replay stores action = (champion_id, position) pairs
        # these need to be converted into the corresponding index of the input vector to the Qnet
        actions = np.array([start.get_action(*exp[1]) for exp in training_batch])
        targetQ = np.array(updates)
        feed_dict = {self.ddq_net.online_ops["input"]:np.stack([exp[0].format_state() for exp in training_batch],axis=0),
                     self.ddq_net.online_ops["actions"]:actions,
                     self.ddq_net.online_ops["target"]:targetQ,
                     self.ddq_net.online_ops["dropout_keep_prob"]:0.5}
        _ = self.ddq_net.sess.run(self.ddq_net.online_ops["update"],feed_dict=feed_dict)

    def validate_model(self, data):
        """
        Validates given model by computing loss and absolute accuracy for data using current Qnet.
        Args:
            data (list(dict)): list of matches to validate against
        Returns:
            stats (tuple(float)): list of statistical measures of performance. stats = (loss,acc)
        """
        buf = []
        for match in data:
            # Loss is only computed for winning side of drafts
            team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM
            # Process match into individual experiences
            experiences = mp.process_match(match, team)
            for exp in experiences:
                _,act,_,_ = exp
                (cid,pos) = act
                if cid is None:
                    # Skip null actions such as missing/skipped bans
                    continue
                buf.append(exp)

        n_exp = len(buf)
        targets = []
        for exp in buf:
            start,_,reward,end = exp
            state_code = end.evaluate()
            if(state_code==DraftState.DRAFT_COMPLETE or state_code in DraftState.invalid_states):
                # Action moves to terminal state
                targets.append(reward)
            else:
                feed_dict = {self.ddq_net.online_ops["input"]:[end.format_state()],
                             self.ddq_net.online_ops["valid_actions"]:[end.get_valid_actions()]}
                predicted_action = self.ddq_net.sess.run(self.ddq_net.online_ops["prediction"], feed_dict=feed_dict)[0]

                feed_dict = {self.ddq_net.target_ops["input"]:[end.format_state()]}
                predicted_Q = self.ddq_net.sess.run(self.ddq_net.target_ops["outQ"], feed_dict=feed_dict)

                targets.append(reward + self.ddq_net.discount_factor*predicted_Q[0,predicted_action])

        actions = np.array([start.get_action(*exp[1]) for exp in buf])
        targets = np.array(targets)

        feed_dict = {self.ddq_net.online_ops["input"]:np.stack([exp[0].format_state() for exp in buf],axis=0),
                     self.ddq_net.online_ops["actions"]:actions,
                     self.ddq_net.online_ops["target"]:targets,
                     self.ddq_net.online_ops["valid_actions"]:np.stack([exp[0].get_valid_actions() for exp in buf],axis=0)}

        loss, pred_q = self.ddq_net.sess.run([self.ddq_net.online_ops["loss"], self.ddq_net.online_ops["valid_outQ"]],feed_dict=feed_dict)

        accurate_predictions = 0
        rank_tolerance = 5
        for n in range(n_exp):
            state,act,_,_ = buf[n]
            submitted_action_id = state.get_action(*act)

            data = [(a,pred_q[n,a]) for a in range(pred_q.shape[1])]
            df = pd.DataFrame(data, columns=['act_id','Q'])
            df.sort_values('Q',ascending=False,inplace=True)
            df.reset_index(drop=True,inplace=True)
            df['rank'] = df.index
            submitted_row = df[df['act_id']==submitted_action_id]
            rank = submitted_row['rank'].iloc[0]
            if rank < rank_tolerance:
                accurate_predictions += 1

        accuracy = accurate_predictions/n_exp
        return (loss, accuracy)

class SoftmaxTrainer(BaseTrainer):
    def __init__(self, network, n_epoch, training_data, validation_data, batch_size, load_path=None):
        num_episodes = len(training_data)
        print("***")
        print("Beginning training..")
        print("  train_epochs: {}".format(n_epoch))
        print("  num_episodes: {}".format(num_episodes))
        print("  batch_size: {}".format(batch_size))
        print("***")

        self.model = network
        self.n_epoch = n_epoch
        self.training_data = training_data
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.load_path = load_path

        self.step_count = 0
        self.epoch_count = 0

        self.teams = [DraftState.BLUE_TEAM, DraftState.RED_TEAM]

        self._buffer = er.ExperienceBuffer(max_buffer_size=20*len(training_data))
        self._val_buffer = er.ExperienceBuffer(max_buffer_size=20*len(validation_data))

        self.fill_buffer(training_data, self._buffer)
        self.fill_buffer(validation_data, self._val_buffer)

    def fill_buffer(self, data, buf):
        for match in data:
            for team in self.teams:
                experiences = mp.process_match(match, team)
                # remove null actions (usually missing bans)
                for exp in experiences:
                    _,act,_,_ = exp
                    cid,pos = act
                    if(cid):
                        buf.store([exp])

    def sample_buffer(self, buf, n_samples):
        experiences = buf.sample(n_samples)
        states = []
        actions = []
        valid_actions = []
        for (state, action, _, _) in experiences:
            states.append(state.format_state())
            valid_actions.append(state.get_valid_actions())
            actions.append(state.get_action(*action))

        return (states, actions, valid_actions)

    def train(self):
        summaries = {}
        summaries["loss"] = []
        summaries["train_acc"] = []
        summaries["val_acc"] = []

        lr_decay_freq = 10
        min_learning_rate = 1.e-8 # Minimum learning rate allowed to decay to

        stash_model = True # Flag for stashing a copy of the model
        model_stash_interval = 10 # Stashes a copy of the model this often

        # Load existing model
        self.model.sess.run(self.model.ops_dict["init"])
        if(self.load_path):
            self.model.load(self.load_path)
            print("\nCheckpoint loaded from {}".format(self.load_path))

        for self.epoch_count in range(self.n_epoch):
            learning_rate = self.model.ops_dict["learning_rate"].eval(self.model.sess)
            if((self.epoch_count>0) and (self.epoch_count % lr_decay_freq == 0) and (learning_rate>= min_learning_rate)):
                # Decay learning rate accoring to schedule
                learning_rate = 0.5*learning_rate
                self.model.sess.run(self.model.ops_dict["learning_rate"].assign(learning_rate))

            t0 =  time.time()
            loss, train_acc, val_acc = self.train_epoch()
            dt = time.time()-t0
            print(" Finished epoch {:2}/{}: lr: {:.4e}, dt {:.2f}, loss {:.6f}, train {:.6f}, val {:.6f}".format(self.epoch_count+1, self.n_epoch, learning_rate, dt, loss, train_acc, val_acc), flush=True)
            summaries["loss"].append(loss)
            summaries["train_acc"].append(train_acc)
            summaries["val_acc"].append(val_acc)

            if(stash_model):
                if(self.epoch_count>0 and (self.epoch_count+1)%model_stash_interval==0):
                    # Stash a copy of the current model
                    out_path = "tmp/models/{}_model_E{}.ckpt".format(self.model._name, self.epoch_count+1)
                    self.model.save(path=out_path)
                    print("Stashed a copy of the current model in {}".format(out_path))

        self.model.save(path=self.model._path_to_model)
        return summaries

    def train_epoch(self):
        n_iter = self._buffer.buffer_size // self.batch_size

        for it in range(n_iter):
            self.train_step()

        loss, train_acc = self.validate_model(self._buffer)
        _, val_acc = self.validate_model(self._val_buffer)

        return (loss, train_acc, val_acc)

    def train_step(self):
        states, actions, valid_actions = self.sample_buffer(self._buffer, self.batch_size)

        feed_dict = {self.model.ops_dict["input"]:np.stack(states, axis=0),
                     self.model.ops_dict["valid_actions"]:np.stack(valid_actions, axis=0),
                     self.model.ops_dict["actions"]:actions,
                     self.model.ops_dict["dropout_keep_prob"]:0.5}
        _  = self.model.sess.run(self.model.ops_dict["update"], feed_dict=feed_dict)

    def validate_model(self, buf):
        states, actions, valid_actions = self.sample_buffer(buf, buf.get_buffer_size())

        feed_dict = {self.model.ops_dict["input"]:np.stack(states, axis=0),
                     self.model.ops_dict["valid_actions"]:np.stack(valid_actions, axis=0),
                     self.model.ops_dict["actions"]:actions}
        loss, train_probs = self.model.sess.run([self.model.ops_dict["loss"], self.model.ops_dict["probabilities"]], feed_dict=feed_dict)

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
