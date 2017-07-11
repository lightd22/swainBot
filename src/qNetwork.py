import tensorflow as tf
import numpy as np

class Qnetwork():
    """
    Args:
        input_size (tuple): tuple of inputs to network.
        output_size (int): number of output nodes for network.
        filter_sizes (tuple of 2 ints): number of filters in each of the two hidden layers. Defaults to (16,32).
        learning_rate (float): network's willingness to change current weights given new example
        discount_factor (float): factor by which future reward after next action is taken are discounted
        regularization (float): strength of weights regularization term in loss function

    A simple Q-network class which is responsible for holding and updating the weights and biases used in predicing Q-values for a given state. This Q-network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into a vector of length input_size.
    2-3) Two fully connected relu-activated hidden layers
    4) Output- linearly activated estimations for Q-values Q(s,a) for each of the output_size actions a available from state s.

    """
    def weight_variable(shape):
        initial = tf.multiply(tf.random_uniform(shape,0,0.1), tf.sqrt(2.0/shape[0]))
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self, inputShape, output_size, filter_sizes = (8,16), learning_rate = 0.001 , discount_factor = 0.9, regularization_coeff = 0.01):
        self.discount_factor = discount_factor
        self.regularization_coeff = regularization_coeff

        # Incoming state matrices are of size input_size = (nChampions, nPos+2)
        # 'None' here means the input tensor will flex with the number of training
        # examples (aka batch size).
        self.input = tf.placeholder(tf.float32, (None,)+inputShape)
        # Reshape input for conv layer to be a tensor of shape [None, nChampions, nPos+2, 1]
        # The extra '1' dimension added at the end represents the number of input channels
        # (normally color channels for an image).
        self.conv_input = tf.expand_dims(self.input,-1)

        self.n_hidden_layers = len(filter_sizes)
        self.n_layers = self.n_hidden_layers + 2

        # First convolutional layer:
        #  32 filters of 3x3 stencil, step size 1, RELu activation, and padding to
        #  keep output spatial shape the same as the input shape
        self.conv1 = tf.layers.conv2d(
                        inputs=self.conv_input,
                        filters=filter_sizes[0],
                        kernel_size=[3,3],
                        padding="SAME",
                        activation=tf.nn.relu,
                        use_bias=True,
                        bias_initializer=tf.constant_initializer(0.1))

        # First pooling layer:
        #  2x2 max pooling with stride 2. Cuts spatial dimensions in half.
        #  Uses padding when input dimensions are odd.
        self.pool1 = tf.layers.max_pooling2d(
                        inputs=self.conv1,
                        pool_size=[2,2],
                        strides=2,
                        padding="SAME")

        # Second convolutional layer:
        #   64 filters of 3x3 stencil, stride 1, RELu activation, and padding to keep
        #   spatial dimensions unchanged
        self.conv2 = tf.layers.conv2d(
                        inputs=self.pool1,
                        filters=filter_sizes[1],
                        kernel_size=[3,3],
                        padding="SAME",
                        activation=tf.nn.relu,
                        use_bias=True,
                        bias_initializer=tf.constant_initializer(0.1))

        # Second pooling layer. Identical parameterization to first pooling layer.
        self.pool2 = tf.layers.max_pooling2d(
                        inputs=self.conv2,
                        pool_size=[2,2],
                        strides=2,
                        padding="SAME")

        # Fully connected (FC) output layer:
        # Flatten input feature map (pool2) to be shape [-1, features]
        # If pool2 has shape = [-1, nx, ny, nf] then feature_size = nx*ny*nf
        self.fc_input_size = int(np.prod(self.pool2.shape[1:]))
        self.pool2_flat = tf.reshape(self.pool2, [-1, self.fc_input_size])
        self.fc_weights = Qnetwork.weight_variable([self.fc_input_size,output_size])
        self.fc_biases = Qnetwork.bias_variable([output_size])
        self.outQ = tf.add(tf.matmul(self.pool2_flat, self.fc_weights), self.fc_biases)

        # Predicted optimal action
        self.prediction = tf.argmax(self.outQ, dimension=1)

        # Loss function and optimization:
        # The inputs self.target and self.actions are indexed by training example. If
        # s[i] = starting state for ith training example (recall that input state s is described by a vector so this will be a matrix)
        # a*[i] = action taken from state s[i] during this training sample
        # Q*(s[i],a*[i]) = the actual value observed from taking action a*[i] from state s[i]
        # outQ[i,-] = estimated values for all actions from state s[i]
        # Then we can write the inputs as
        # self.target[i] = Q*(s[i],a*[i])
        # self.actions[i] = a*[i]

        self.target = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.int32, shape=[None])

        # Since the Qnet outputs a vector Q(s,-) of  predicted values for every possible action that can be taken from state s,
        # we need to connect each target value with the appropriate predicted Q(s,a*) = Qout[i,a*[i]].
        # For some reason this isn't easy for tensorflow to do. So we must manually form the list of
        # [i, actions[i]] index pairs for outQ..
        self.ind = tf.stack([tf.range(tf.shape(self.actions)[0]),self.actions],axis=1)
        # and then "gather" them.
        self.estimatedQ = tf.gather_nd(self.outQ, self.ind)

        # Simple sum-of-squares loss (error) function with regularization. Note that biases do not
        # need to be regularized since they are (generally) not subject to overfitting.
        self.loss = (tf.reduce_mean(tf.square(self.target-self.estimatedQ))+
                    self.regularization_coeff*(tf.nn.l2_loss(self.fc_weights)))

        self.trainer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.updateModel = self.trainer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
