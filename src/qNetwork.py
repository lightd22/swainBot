import tensorflow as tf
import numpy as np

class Qnetwork():
    """
    Args:
        input_shape (tuple): tuple of inputs to network.
        output_shape (int): number of output nodes for network.
        filter_sizes (tuple of 2 ints): number of filters in each of the two hidden layers. Defaults to (16,32).
        learning_rate (float): network's willingness to change current weights given new example
        discount_factor (float): factor by which future reward after next action is taken are discounted
        regularization (float): strength of weights regularization term in loss function

    A simple Q-network class which is responsible for holding and updating the weights and biases used in predicing Q-values for a given state. This Q-network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into an [n_batch, *input_shape] tensor.
    2-4) Three layers of relu-activated hidden 2d convolutional layers with max pooling
    4) Output- linearly activated estimations for Q-values Q(s,a) for each of the output_shape actions a available.

    """
    @property
    def name(self):
        return self._name

    @property
    def discount_factor(self):
        return self._discount_factor

    def weight_variable(shape,name):
        initial = tf.multiply(tf.random_normal(shape,0,1.0), tf.sqrt(2.0/shape[0]))
        return tf.Variable(initial,name)

    def bias_variable(shape,name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial,name)

    def __init__(self, name, input_shape, output_shape, filter_sizes = (8,16,32), learning_rate=1.e-3, regularization_coeff = 0.01, discount_factor = 0.9):
        self._name = name
        self._input_shape = input_shape
        self._secondary_input_shape = (6,)
        self._output_shape = output_shape
        self._regularization_coeff = regularization_coeff
        self._discount_factor = discount_factor
        self._n_hidden_layers = len(filter_sizes)
        self._n_layers = self._n_hidden_layers + 2

        with tf.variable_scope(self._name):
            self.learning_rate = tf.Variable(learning_rate,trainable=False, name="learning_rate")

            # Incoming state matrices are of size input_size = (nChampions, nPos+2)
            # 'None' here means the input tensor will flex with the number of training
            # examples (aka batch size).
            self.input = tf.placeholder(tf.float32, (None,)+input_shape, name="inputs")
            # Reshape input for conv layer to be a tensor of shape [None, nChampions, nPos+2, 1]
            # The extra '1' dimension added at the end represents the number of input channels
            # (normally color channels for an image).
            conv_input = tf.expand_dims(self.input,-1)

            # First convolutional layer:
            #  filters of 3x3 stencil, step size 1, RELu activation, and padding to
            #  keep output spatial shape the same as the input shape
            self.conv1 = tf.layers.conv2d(
                            inputs=conv_input,
                            filters=filter_sizes[0],
                            kernel_size=[3,3],
                            padding="SAME",
                            activation=tf.nn.relu,
                            use_bias=True,
                            bias_initializer=tf.constant_initializer(0.1),
                            name="conv1")

            # First pooling layer:
            #  2x2 max pooling with stride 2. Cuts spatial dimensions in half.
            #  Uses padding when input dimensions are odd.
            self.pool1 = tf.layers.max_pooling2d(
                            inputs=self.conv1,
                            pool_size=[2,2],
                            strides=2,
                            padding="SAME",
                            name="pool1")

            # Second convolutional layer:
            #   filters of 3x3 stencil, stride 1, RELu activation, and padding to keep
            #   spatial dimensions unchanged
            self.conv2 = tf.layers.conv2d(
                            inputs=self.pool1,
                            filters=filter_sizes[1],
                            kernel_size=[3,3],
                            padding="SAME",
                            activation=tf.nn.relu,
                            use_bias=True,
                            bias_initializer=tf.constant_initializer(0.1),
                            name="conv2")

            # Second pooling layer. Identical parameterization to other pooling layers.
            self.pool2 = tf.layers.max_pooling2d(
                            inputs=self.conv2,
                            pool_size=[2,2],
                            strides=2,
                            padding="SAME",
                            name="pool2")

            # Third convolutional layer.
            self.conv3 = tf.layers.conv2d(
                            inputs=self.pool2,
                            filters=filter_sizes[2],
                            kernel_size=[3,3],
                            padding="SAME",
                            activation=tf.nn.relu,
                            use_bias=True,
                            bias_initializer=tf.constant_initializer(0.1),
                            name="conv3")

            # Third pooling layer. Identical parameterization to other pooling layers.
            self.pool3 = tf.layers.max_pooling2d(
                            inputs=self.conv3,
                            pool_size=[2,2],
                            strides=2,
                            padding="SAME",
                            name="pool3")

            # Fully connected (FC) layer:
            # Flatten input feature map (pool2) to be shape [-1, feature_size]
            # If pool3 has shape = [-1, nx, ny, nf] then feature_size = nx*ny*nf
            dim = int(np.prod(self.pool3.shape[1:]))
            pool3_flat = tf.reshape(self.pool3, [-1, dim])
            self.dropout_keep_prob = tf.placeholder_with_default(1.0,shape=())
            self.dropout1 = tf.nn.dropout(pool3_flat,self.dropout_keep_prob)
            self.secondary_input = tf.placeholder(tf.float32, (None,)+self._secondary_input_shape, name="secondary_inputs")

            self.fc_input = tf.concat([self.dropout1, self.secondary_input],axis=1)
            #self.fc_input = tf.concat([pool3_flat,self.secondary_input],axis=1)
            fc1_input_size = int(self.fc_input.shape[1])
            fc1_output_size = fc1_input_size//2

            self.fc1_weights = Qnetwork.weight_variable([fc1_input_size,fc1_output_size],"fc1_weight")
            self.fc1_biases = Qnetwork.bias_variable([fc1_output_size],"fc1_bias")

            self.fc1 = tf.nn.relu(tf.add(tf.matmul(self.fc_input, self.fc1_weights), self.fc1_biases),name="fc1")
            self.dropout2 = tf.nn.dropout(self.fc1,self.dropout_keep_prob)

            # FC output layer
            self.fc2_weights = Qnetwork.weight_variable([fc1_output_size,output_shape],"fc2_weight")
            self.fc2_biases = Qnetwork.bias_variable([output_shape],"fc2_bias")
            self.outQ = tf.add(tf.matmul(self.dropout2, self.fc2_weights), self.fc2_biases, name="outputs")

            # Predicted optimal action
            self.valid_actions = tf.placeholder(tf.bool, shape=self.outQ.shape, name="valid_actions")
            self.prediction  = tf.argmax(tf.where(self.valid_actions, self.outQ, tf.scalar_mul(-np.inf,tf.ones_like(self.outQ))), axis=1, name="prediction")
            #self.prediction = tf.argmax(self.outQ, axis=1, name="prediction")

            # Loss function and optimization:
            # The inputs self.target and self.actions are indexed by training example. If
            # s[i] = starting state for ith training example (recall that input state s is described by a vector so this will be a matrix)
            # a*[i] = action taken from state s[i] during this training sample
            # Q*(s[i],a*[i]) = the actual value observed from taking action a*[i] from state s[i]
            # outQ[i,-] = estimated values for all actions from state s[i]
            # Then we can write the inputs as
            # self.target[i] = Q*(s[i],a*[i])
            # self.actions[i] = a*[i]

            self.target = tf.placeholder(tf.float32, shape=[None], name="target_Q")
            self.actions = tf.placeholder(tf.int32, shape=[None], name="submitted_action")

            # Since the Qnet outputs a vector Q(s,-) of  predicted values for every possible action that can be taken from state s,
            # we need to connect each target value with the appropriate predicted Q(s,a*) = Qout[i,a*[i]].
            # Main idea is to get indexes into the outQ tensor based on input actions and gather the resulting Q values
            # For some reason this isn't easy for tensorflow to do. So we must manually form the list of
            # [i, actions[i]] index pairs for outQ..
            #  n_batch = outQ.shape[0] = actions.shape[0]
            #  n_actions = outQ.shape[1]
            ind = tf.stack([tf.range(tf.shape(self.actions)[0]),self.actions],axis=1)
            # and then "gather" them.
            self.estimatedQ = tf.gather_nd(self.outQ, ind)
            # Special notes: this is more efficient than indexing into the flattened version of outQ (which I have seen before)
            # because the gather operation is applied to outQ directly. Apparently this propagates the gradient more efficiently
            # under specific sparsity conditions (which tf.Variables like outQ satisfy)

            # Simple sum-of-squares loss (error) function with regularization. Note that biases do not
            # need to be regularized since they are (generally) not subject to overfitting.
            self.loss = tf.add(tf.reduce_mean(0.5*tf.square(self.target-self.estimatedQ)),
                        self._regularization_coeff*(tf.nn.l2_loss(self.fc1_weights)+tf.nn.l2_loss(self.fc2_weights)),name="loss")

            self.trainer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.update = self.trainer.minimize(self.loss, name="update")

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
