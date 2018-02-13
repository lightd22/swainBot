import tensorflow as tf
from . import base_model

class QNetInferenceModel(base_model.BaseModel):
    def __init__(self, name, path):
        super().__init__(name=name, path=path)
        self.init_saver()
        self.ops_dict = self.build_model()

    def init_saver(self):
        with self._graph.as_default():
            self.saver = tf.train.import_meta_graph("{path}.ckpt.meta".format(path=self._path_to_model))
            self.saver.restore(self.sess,"{path}.ckpt".format(path=self._path_to_model))
    def build_model(self):
        ops_dict = {}
        with self._graph.as_default():
            ops_dict["predict_q"] = tf.get_default_graph().get_tensor_by_name("online/valid_q_vals:0")
            ops_dict["prediction"] = tf.get_default_graph().get_tensor_by_name("online/prediction:0")
            ops_dict["input"] = tf.get_default_graph().get_tensor_by_name("online/inputs:0")
            ops_dict["valid_actions"] = tf.get_default_graph().get_tensor_by_name("online/valid_actions:0")
        return ops_dict

    def predict(self, states):
        """
        Feeds state into model and returns current predicted Q-values.
        Args:
            states (list of DraftStates): states to predict from
        Returns:
            predicted_Q (numpy array): model estimates of Q-values for actions from input states.
              predicted_Q[k,:] holds Q-values for state states[k]
        """
        inputs = [state.format_state() for state in states]
        valid_actions = [state.get_valid_actions() for state in states]

        feed_dict = {self.ops_dict["input"]:inputs,
                     self.ops_dict["valid_actions"]:valid_actions}
        predicted_Q = self.sess.run(self.ops_dict["predict_q"], feed_dict=feed_dict)
        return predicted_Q

    def predict_action(self, states):
        """
        Feeds state into model and return recommended action to take from input state based on estimated Q-values.
        Args:
            state (list of DraftStates): states to predict from
        Returns:
            predicted_action (numpy array): array of integer representations of actions recommended by model.
        """
        inputs = [state.format_state() for state in states]
        valid_actions = [state.get_valid_actions() for state in states]

        feed_dict = {self.ops_dict["input"]:inputs,
                     self.ops_dict["valid_actions"]:valid_actions}
        predicted_actions = self.sess.run(self.ops_dict["prediction"], feed_dict=feed_dict)
        return predicted_actions

class SoftmaxInferenceModel(base_model.BaseModel):
    def __init__(self, name, path):
        super().__init__(name=name, path=path)
        self.init_saver()
        self.ops_dict = self.build_model()

    def init_saver(self):
        with self._graph.as_default():
            self.saver = tf.train.import_meta_graph("{path}.ckpt.meta".format(path=self._path_to_model))
            self.saver.restore(self.sess,"{path}.ckpt".format(path=self._path_to_model))

    def build_model(self):
        ops_dict = {}
        with self._graph.as_default():
            ops_dict["probabilities"] = tf.get_default_graph().get_tensor_by_name("softmax/action_probabilites:0")
            ops_dict["prediction"] = tf.get_default_graph().get_tensor_by_name("softmax/predictions:0")
            ops_dict["input"] = tf.get_default_graph().get_tensor_by_name("softmax/inputs:0")
            ops_dict["valid_actions"] = tf.get_default_graph().get_tensor_by_name("softmax/valid_actions:0")
        return ops_dict

    def predict(self, states):
        """
        Feeds state into model and returns current predicted probabilities.
        Args:
            states (list of DraftStates): states to predict from
        Returns:
            probabilities (numpy array): model estimates of probabilities for actions from input states.
              probabilities[k,:] holds Q-values for state states[k]
        """
        inputs = [state.format_state() for state in states]
        valid_actions = [state.get_valid_actions() for state in states]

        feed_dict = {self.ops_dict["input"]:inputs,
                     self.ops_dict["valid_actions"]:valid_actions}
        probabilities = self.sess.run(self.ops_dict["probabilities"], feed_dict=feed_dict)
        return probabilities

    def predict_action(self, states):
        """
        Feeds state into model and return recommended action to take from input state based on estimated Q-values.
        Args:
            state (list of DraftStates): states to predict from
        Returns:
            predicted_action (numpy array): array of integer representations of actions recommended by model.
        """
        inputs = [state.format_state() for state in states]
        valid_actions = [state.get_valid_actions() for state in states]

        feed_dict = {self.ops_dict["input"]:inputs,
                     self.ops_dict["valid_actions"]:valid_actions}
        predicted_actions = self.sess.run(self.ops_dict["prediction"], feed_dict=feed_dict)
        return predicted_actions
