import tensorflow as tf
import numpy as np

from . import base_model

class Qnetwork(base_model.BaseModel):
    """
    Args:
        name (string): label for model namespace
        path (string): path to save/load model
        input_shape (tuple): tuple of inputs to network.
        output_shape (int): number of output nodes for network.
        filter_sizes (tuple of ints): number of filters in each of the two hidden layers. Defaults to (512,512).
        learning_rate (float): network's willingness to change current weights given new example
        regularization (float): strength of weights regularization term in loss function
        discount_factor (float): factor by which future reward after next action is taken are discounted
        tau (float): Hyperparameter used in updating target network (if used)
             Some notable values:
              tau = 1.e-3 -> used in original paper
              tau = 0.5 -> average DDQN
              tau = 1.0 -> copy online -> target

    A Q-network class which is responsible for holding and updating the weights and biases used in predicing Q-values for a given state. This Q-network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into an [n_batch, *input_shape] tensor.
    2) Two layers of relu-activated hidden fc layers with dropout
    3) Output- linearly activated estimations for Q-values Q(s,a) for each of the output_shape actions a available.

    """
    @property
    def name(self):
        return self._name

    @property
    def discount_factor(self):
        return self._discount_factor

    def __init__(self, name, path, input_shape, output_shape, filter_sizes=(512,512), learning_rate=1.e-5, regularization_coeff=1.e-4, discount_factor=0.9, tau=1.0):
        super().__init__(name=name, path=path)
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._filter_sizes = filter_sizes
        self._learning_rate = learning_rate
        self._regularization_coeff = regularization_coeff
        self._discount_factor = discount_factor
        self._n_hidden_layers = len(filter_sizes)
        self._n_layers = self._n_hidden_layers + 2
        self._tau = tau

        self.online_name = "online"
        self.target_name = "target"
        # Build base Q-network model
        self.online_ops = self.build_model(name = self.online_name)
        # If using a target network for DDQN network, add related ops to model
        if(self.target_name):
            self.target_ops = self.build_model(name = self.target_name)
            self.target_ops["target_init"] = self.create_target_initialization_ops(self.target_name, self.online_name)
            self.target_ops["target_update"] = self.create_target_update_ops(self.target_name, self.online_name, tau=self._tau)
        with self._graph.as_default():
            self.online_ops["init"] = tf.global_variables_initializer()
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

                # FC output layer
                ops_dict["outQ"] = tf.layers.dense(
                    dropout1,
                    self._output_shape,
                    activation=None,
                    bias_initializer=tf.constant_initializer(0.1),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self._regularization_coeff),
                    name="q_vals")

                # Placeholder for valid actions filter
                ops_dict["valid_actions"] = tf.placeholder(tf.bool, shape=ops_dict["outQ"].shape, name="valid_actions")

                # Filtered Q-values
                ops_dict["valid_outQ"] = tf.where(ops_dict["valid_actions"], ops_dict["outQ"], tf.scalar_mul(-np.inf,tf.ones_like(ops_dict["outQ"])), name="valid_q_vals")

                # Max Q value amongst valid actions
                ops_dict["max_Q"] = tf.reduce_max(ops_dict["valid_outQ"], axis=1, name="max_Q")

                # Predicted optimal action amongst valid actions
                ops_dict["prediction"]  = tf.argmax(ops_dict["valid_outQ"], axis=1, name="prediction")

                # Loss function and optimization:
                # The inputs self.target and self.actions are indexed by training example. If
                # s[i] = starting state for ith training example (recall that input state s is described by a vector so this will be a matrix)
                # a*[i] = action taken from state s[i] during this training sample
                # Q*(s[i],a*[i]) = the actual value observed from taking action a*[i] from state s[i]
                # outQ[i,-] = estimated values for all actions from state s[i]
                # Then we can write the inputs as
                # self.target[i] = Q*(s[i],a*[i])
                # self.actions[i] = a*[i]

                ops_dict["target"] = tf.placeholder(tf.float32, shape=[None], name="target_Q")
                ops_dict["actions"] = tf.placeholder(tf.int32, shape=[None], name="submitted_action")

                # Since the Qnet outputs a vector Q(s,-) of  predicted values for every possible action that can be taken from state s,
                # we need to connect each target value with the appropriate predicted Q(s,a*) = Qout[i,a*[i]].
                # Main idea is to get indexes into the outQ tensor based on input actions and gather the resulting Q values
                # For some reason this isn't easy for tensorflow to do. So we must manually form the list of
                # [i, actions[i]] index pairs for outQ..
                #  n_batch = outQ.shape[0] = actions.shape[0]
                #  n_actions = outQ.shape[1]
                ind = tf.stack([tf.range(tf.shape(ops_dict["actions"])[0]),ops_dict["actions"]],axis=1)
                # and then "gather" them.
                estimatedQ = tf.gather_nd(ops_dict["outQ"], ind)
                # Special notes: this is more efficient than indexing into the flattened version of outQ (which I have seen before)
                # because the gather operation is applied to outQ directly. Apparently this propagates the gradient more efficiently
                # under specific sparsity conditions (which tf.Variables like outQ satisfy)

                # Simple sum-of-squares loss (error) function. Note that biases do not
                # need to be regularized since they are (generally) not subject to overfitting.
                ops_dict["loss"] = tf.reduce_mean(0.5*tf.square(ops_dict["target"]-estimatedQ), name="loss")

                ops_dict["trainer"] = tf.train.AdamOptimizer(learning_rate = ops_dict["learning_rate"])
                ops_dict["update"] = ops_dict["trainer"].minimize(ops_dict["loss"], name="update")

        return ops_dict

    def create_target_update_ops(self, target_scope, online_scope, tau=1e-3, name="target_update"):
        """
        Adds operations to graph which are used to update the target network after after a training batch is sent
        through the online network.

        This function should be executed only once before training begins. The resulting operations should
        be run within a tf.Session() once per training batch.

        In double-Q network learning, the online (primary) network is updated using traditional backpropegation techniques
        with target values produced by the target-Q network.
        To improve stability, the target-Q is updated using a linear combination of its current weights
        with the current weights of the online network:
            Q_target = tau*Q_online + (1-tau)*Q_target
        Typical tau values are small (tau ~ 1e-3). For more, see https://arxiv.org/abs/1509.06461 and https://arxiv.org/pdf/1509.02971.pdf.
        Args:
            target_scope (str): name of scope that target network occupies
            online_scope (str): name of scope that online network occupies
            tau (float32): Hyperparameter for combining target-Q and online-Q networks
            name (str): name of operation which updates the target network when run within a session
        Returns: Tensorflow operation which updates the target nework when run.
        """
        with self._graph.as_default():
            target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope)
            online_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=online_scope)
            ops = [target_params[i].assign(tf.add(tf.multiply(tau,online_params[i]),tf.multiply(1.-tau,target_params[i]))) for i in range(len(target_params))]
            return tf.group(*ops,name=name)

    def create_target_initialization_ops(self, target_scope, online_scope):
        """
        This adds operations to the graph in order to initialize the target Q network to the same values as the
        online network.

        This function should be executed only once just after the online network has been initialized.

        Args:
            target_scope (str): name of scope that target network occupies
            online_scope (str): name of scope that online network occupies
        Returns:
            Tensorflow operation (named "target_init") which initialize the target nework when run.
        """
        return self.create_target_update_ops(target_scope, online_scope, tau=1.0, name="target_init")
