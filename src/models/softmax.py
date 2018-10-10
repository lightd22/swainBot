import tensorflow as tf
import numpy as np

from . import base_model

class SoftmaxNetwork(base_model.BaseModel):
    """
    Args:
        input_shape (tuple): tuple of inputs to network.
        output_shape (int): number of output nodes for network.
        filter_sizes (tuple of 2 ints): number of filters in each of the two hidden layers. Defaults to (16,32).
        learning_rate (float): network's willingness to change current weights given new example
        regularization (float): strength of weights regularization term in loss function

    A simple softmax network class which is responsible for holding and updating the weights and biases used in predicing actions for given state. This network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into an [n_batch, *input_shape] tensor.
    2-4) Two layers of relu-activated hidden fc layers
    4) Output- softmax-obtained probability of action submission for output_shape actions available.

    """
    @property
    def name(self):
        return self._name

    def __init__(self, name, path, input_shape, output_shape, filter_sizes = (512,512), learning_rate=1.e-3, regularization_coeff = 0.01):
        super().__init__(name=name, path=path)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._learning_rate = learning_rate
        self._regularization_coeff = regularization_coeff
        self._n_hidden_layers = len(filter_sizes)
        self._n_layers = self._n_hidden_layers + 2
        self._filter_sizes = filter_sizes

        self.ops_dict = self.build_model(name=self._name)
        with self._graph.as_default():
            self.ops_dict["init"] = tf.global_variables_initializer()

        self.init_saver()

    def init_saver(self):
        with self._graph.as_default():
            self.saver = tf.train.Saver()

    def save(self, path):
        self.saver.save(self.sess, save_path=path)

    def load(self, path):
        self.saver.restore(self.sess, save_path=path)

    def build_model(self, name):
        ops_dict = {}
        with self._graph.as_default():
            with tf.variable_scope(name):
                ops_dict["learning_rate"] = tf.Variable(self._learning_rate, trainable=False, name="learning_rate")

                # Incoming state matrices are of size input_size = (nChampions, nPos+2)
                # 'None' here means the input tensor will flex with the number of training
                # examples (aka batch size).
                ops_dict["input"] = tf.placeholder(tf.float32, (None,)+self._input_shape, name="inputs")
                ops_dict["dropout_keep_prob"] = tf.placeholder_with_default(1.0,shape=())

                # Fully connected (FC) layers:
                fc0 = tf.layers.dense(
                    ops_dict["input"],
                    self._filter_sizes[0],
                    activation=tf.nn.relu,
                    bias_initializer=tf.constant_initializer(0.1),
                    name="fc_0")
                dropout0 = tf.nn.dropout(fc0, ops_dict["dropout_keep_prob"])

                fc1 = tf.layers.dense(
                    dropout0,
                    self._filter_sizes[1],
                    activation=tf.nn.relu,
                    bias_initializer=tf.constant_initializer(0.1),
                    name="fc_1")
                dropout1 = tf.nn.dropout(fc1, ops_dict["dropout_keep_prob"])

                # Logits layer
                ops_dict["logits"] = tf.layers.dense(
                    dropout1,
                    self._output_shape,
                    activation=None,
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._regularization_coeff),
                    name="logits")

                # Placeholder for valid actions filter
                ops_dict["valid_actions"] = tf.placeholder(tf.bool, shape=ops_dict["logits"].shape, name="valid_actions")

                # Filtered logits
                ops_dict["valid_logits"] = tf.where(ops_dict["valid_actions"], ops_dict["logits"], tf.scalar_mul(-np.inf, tf.ones_like(ops_dict["logits"])), name="valid_logits")

                # Predicted optimal action amongst valid actions
                ops_dict["probabilities"]  = tf.nn.softmax(ops_dict["valid_logits"], name="action_probabilites")
                ops_dict["prediction"] = tf.argmax(input=ops_dict["valid_logits"], axis=1, name="predictions")

                ops_dict["actions"] = tf.placeholder(tf.int32, shape=[None], name="submitted_actions")

                ops_dict["loss"] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ops_dict["actions"], logits=ops_dict["valid_logits"]), name="loss")

                ops_dict["trainer"] = tf.train.AdamOptimizer(learning_rate = ops_dict["learning_rate"])
                ops_dict["update"] = ops_dict["trainer"].minimize(ops_dict["loss"], name="update")

        return ops_dict
