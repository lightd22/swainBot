import tensorflow as tf
import numpy as np

class SoftmaxNetwork():
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

    def weight_variable(shape,name):
        initial = tf.multiply(tf.random_normal(shape,0,1.0), tf.sqrt(2.0/shape[0]))
        return tf.Variable(initial,name)

    def bias_variable(shape,name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name)

    def __init__(self, name, input_shape, output_shape, filter_sizes = (512,512), learning_rate=1.e-3, regularization_coeff = 0.01):
        self._name = name
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._regularization_coeff = regularization_coeff
        self._learning_rate = learning_rate
        self._n_hidden_layers = len(filter_sizes)
        self._n_layers = self._n_hidden_layers + 2
        self._filter_sizes = filter_sizes

        self.build_model()
        self.init_saver()

    def save(self, sess, path):
        self.saver.save(sess, save_path=path)

    def load(self, sess, path):
        self.saver.restore(sess, save_path=path)

    def build_model(self):
        with tf.variable_scope(self._name):
            self.learning_rate = tf.Variable(self._learning_rate, trainable=False, name="learning_rate")

            # Incoming state matrices are of size input_size = (nChampions, nPos+2)
            # 'None' here means the input tensor will flex with the number of training
            # examples (aka batch size).
            self.input = tf.placeholder(tf.float32, (None,)+self._input_shape, name="inputs")
            self.dropout_keep_prob = tf.placeholder_with_default(1.0,shape=())

            # Fully connected (FC) layers:
            self.fc0 = tf.layers.dense(
                self.input,
                self._filter_sizes[0],
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.1),
                name="fc_0")
            self.dropout0 = tf.nn.dropout(self.fc0, self.dropout_keep_prob)

            self.fc1 = tf.layers.dense(
                self.dropout0,
                self._filter_sizes[1],
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.1),
                name="fc_1")
            self.dropout1 = tf.nn.dropout(self.fc1, self.dropout_keep_prob)

            # Logits layer
            self.logits = tf.layers.dense(
                self.dropout1,
                self._output_shape,
                activation=None,
                bias_initializer=tf.constant_initializer(0.1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._regularization_coeff),
                name="logits")

            # Placeholder for valid actions filter
            self.valid_actions = tf.placeholder(tf.bool, shape=self.logits.shape, name="valid_actions")

            # Filtered logits
            self.valid_logits = tf.where(self.valid_actions, self.logits, tf.scalar_mul(-np.inf, tf.ones_like(self.logits)), name="valid_logits")

            # Predicted optimal action amongst valid actions
            self.probabilities  = tf.nn.softmax(self.valid_logits, name="action_probabilites")
            self.prediction = tf.argmax(input=self.valid_logits, axis=1, name="predictions")

            self.actions = tf.placeholder(tf.int32, shape=[None], name="submitted_actions")

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.valid_logits), name="loss")

            self.trainer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.update = self.trainer.minimize(self.loss, name="update")


        self.init = tf.global_variables_initializer()

    def init_saver(self):
        self.saver = tf.train.Saver()
