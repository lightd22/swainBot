import tensorflow as tf

class Qnetwork():
    """
    Args:
        inputSize (int): number of inputs to network.
        outputSize (int): number of output nodes for network.
        layerSizes (tuple of 2 ints): number of nodes in each of the two hidden layers. Defaults to (280,280).
        learningRate (float): network's willingness to change current weights given new example
        discountFactor (float): factor by which future reward after next action is taken are discounted

    A simple Q-network class which is responsible for holding and updating the weights and biases used in predicing Q-values for a given state. This Q-network will consist of
    the following layers:
    1) Input- a DraftState state s (an array of bool) representing the current state reshaped into a vector of length inputSize.
    2-3) Two fully connected tanh-activated hidden layers
    4) Output- linearly activated estimations for Q-values Q(s,a) for each of the outputSize actions a available from state s.

    """

    def __init__(self, inputSize, outputSize, layerSizes = (280,280), learningRate = 0.001 , discountFactor = 0.9):
        self.discountFactor = discountFactor
        # Input is shape [None, inputSize]. The 'None' means the input tensor will flex
        # with the number of training examples (aka batch size). Each training example will be a row vector of length inputSize.

        self.input = tf.placeholder(tf.float32, [None, inputSize])

        # Set weight and bias dictonaries.
        self.weights = {
            "layer1": tf.Variable(tf.random_normal([inputSize,layerSizes[0]])),
            "layer2": tf.Variable(tf.random_normal([layerSizes[0],layerSizes[1]])),
            "out": tf.Variable(tf.random_normal([layerSizes[1],outputSize]))
        }

        self.biases = {
            "layer1": tf.Variable(tf.random_normal([layerSizes[0]])),
            "layer2": tf.Variable(tf.random_normal([layerSizes[1]])),
            "out": tf.Variable(tf.random_normal([outputSize]))
        }

        # First hidden layer.
        self.layer1 = tf.add(tf.matmul(self.input, self.weights["layer1"]), self.biases["layer1"])
        self.layer1 = tf.nn.tanh(self.layer1)

        # Second hidden layer.
        self.layer2 = tf.add(tf.matmul(self.layer1, self.weights["layer2"]), self.biases["layer2"])
        self.layer2 = tf.nn.tanh(self.layer2)

        # Output layer.
        self.outQ = tf.matmul(self.layer2,self.weights["out"])+self.biases["out"]
        self.prediction = tf.argmax(self.outQ, dimension=1) # Predicted optimal action

        # Loss function and optimization:
        # The inputs self.target and self.actions are indexed by training example. If
        # s[i] = starting state for ith training example (recall that each s is described by a vector so this will be a matrix)
        # a*[i] = action taken from state s[i] during this training sample
        # Q*(s[i],a*[i]) = the actual value observed from taking action a*[i] from state s[i]
        # Then we can write the inputs as
        # self.target[i] = Q*(s[i],a*[i])
        # self.actions[i] = a*[i]

        self.target = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.int32, shape=[None])

        # Since the Qnet outputs a vector Q(s,-) of  predicted values for every possible action that can be taken from state s,
        # we need to connect each self.target value with the appropriate predicted Q(s,a*). This is done with a one_hot representation of
        # self.action. Each row in actionsOneHot[i,:] is a one hot vector encoded by a*[i].
        self.actionsOneHot = tf.one_hot(self.actions, outputSize, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.mul(self.outQ, self.actionsOneHot), 1) # Vector of predicted Q(s[i],a*[i]) values

        # Simple sum-of-squares loss (error) function
        self.loss = tf.reduce_sum(tf.square(self.target-self.Q))

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate = learningRate)
        self.updateModel = self.trainer.minimize(self.loss)

        self.init = tf.initialize_all_variables()
