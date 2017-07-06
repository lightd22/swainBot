import tensorflow as tf

class Qnetwork():
    """
    Args:
        inputSize (int): number of inputs to network.
        outputSize (int): number of output nodes for network.
        layerSizes (tuple of 2 ints): number of nodes in each of the two hidden layers. Defaults to (280,280).
        learningRate (float): network's willingness to change current weights given new example
        discountFactor (float): factor by which future reward after next action is taken are discounted
        regularization (float): strength of weights regularization term in loss function

    A simple Q-network class which is responsible for holding and updating the weights and biases used in predicing Q-values for a given state. This Q-network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into a vector of length inputSize.
    2-3) Two fully connected relu-activated hidden layers
    4) Output- linearly activated estimations for Q-values Q(s,a) for each of the outputSize actions a available from state s.

    """
    def weight_variable(shape):
        initial = tf.multiply(tf.random_uniform(shape,0,0.1), tf.sqrt(2.0/shape[0]))
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self, inputSize, outputSize, layerSizes = (280,280), learningRate = 0.001 , discountFactor = 0.9, regularizationCoeff = 0.01):
        self.discountFactor = discountFactor
        self.regularizationCoeff = regularizationCoeff
        # Input is shape [None, inputSize]. 'None' means the input tensor will flex
        # with the number of training examples (aka batch size). Each training example will be a row vector of length inputSize.

        self.input = tf.placeholder(tf.float32, [None, inputSize])
        self.n_hidden_layers = len(layerSizes)
        self.n_layers = self.n_hidden_layers + 2
        # Initialize weight and bias dictonaries.
        self.weights = {
            1: Qnetwork.weight_variable([inputSize,layerSizes[0]]),
            2: Qnetwork.weight_variable([layerSizes[0],layerSizes[1]]),
            3: Qnetwork.weight_variable([layerSizes[1],outputSize])
        }

        self.biases = {
            1: Qnetwork.bias_variable([layerSizes[0]]),
            2: Qnetwork.bias_variable([layerSizes[1]]),
            3: Qnetwork.bias_variable([outputSize])
        }

        # First hidden layer.
        self.layer1 = tf.add(tf.matmul(self.input, self.weights[1]), self.biases[1])
        self.layer1 = tf.nn.relu(self.layer1)

        # Second hidden layer.
        self.layer2 = tf.add(tf.matmul(self.layer1, self.weights[2]), self.biases[2])
        self.layer2 = tf.nn.relu(self.layer2)

        # Output layer.
        self.outQ = tf.matmul(self.layer2,self.weights[3])+self.biases[3]
        self.prediction = tf.argmax(self.outQ, dimension=1) # Predicted optimal action

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
                    self.regularizationCoeff*(tf.nn.l2_loss(self.weights[1])+
                    tf.nn.l2_loss(self.weights[2])+
                    tf.nn.l2_loss(self.weights[3])))

        self.trainer = tf.train.AdamOptimizer(learning_rate = learningRate)
        self.updateModel = self.trainer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
