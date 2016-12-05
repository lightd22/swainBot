import numpy as np
import tensorflow as tf
import random
from cassiopeia import riotapi
from draftstate import DraftState

import championinfo as cinfo
from rewards import getReward
import matchProcessing as mp
import experienceReplay as er



def trainNetwork(Qnet, numEpisodes, batchSize, bufferSize):
    """
    Args:
        Qnet (qNetwork): Q-network to be trained.
        numEpisodes (int): total number of drafts to be simulated
        batchSize (int): size of each training set sampled from the replay buffer which will be used to update Qnet at a time
        bufferSize (int): size of replay buffer used 
    Returns:
        None
    Trains the Q-network Qnet in batches using experience replays.
    """
    # Initialize tensorflow variables
    init = tf.initialize_all_variables()

    # Initialize experience replay buffer
    experienceReplay = er.ExperienceBuffer(bufferSize)
    
    # For now, we are training off a single draft (my most recent). Later on this will be populated with numEpisode drafts 
    matchQueue = mp.buildMatchQueue()
    matchRef = matchQueue.get()
    match = matchRef.match()

    totalSteps = 0
    # Number of steps to take before doing any training. Needs to be at least batchSize to avoid error when sampling from experience replay
    preTrainingSteps = batchSize
    # Number of steps to take between training
    updateFreq = 3 # 3 -> train after every match

    # Start training
    with tf.Session() as sess:
        sess.run(init)
        for episode in range(numEpisodes):
            # Get next match from queue
            # Devin: For now we are just repeatedly training from a single match for testing purposes
            # matchRef = matchQueue.get()
            # match = matchRef.match()

            team = DraftState.RED_TEAM if match.red_team.win else DraftState.BLUE_TEAM # For now only learn from winning team
            # Add this match to experience replay
            experiences = mp.processMatch(matchRef, team, mode = "ban")
            for experience in experiences:
                experienceReplay.store([experience])
                totalSteps += 1

                # Every updateFreq steps we train the network using the replay buffer
                if (totalSteps > preTrainingSteps) and (totalSteps % updateFreq == 0):
                    trainingBatch = experienceReplay.sample(batchSize)
                    
                    # Each row in predictedQ gives estimated Q(s',a) values for each possible action for a single input state s'. 
                    predictedQ = sess.run(Qnet.outQ,
                                          feed_dict={Qnet.input:np.vstack([exp[3].formatState() for exp in trainingBatch])})
                    
                    # To get max_{a} Q(s',a) values for each s' (required when calculating targetQ), take max along *rows* of predictedQ.
                    maxQ = np.max(predictedQ,axis=1)

                    # Calculate target Q values for each example
                    targetQ = trainingBatch[:,2] + Qnet.discountFactor*maxQ[:]

                    # Update Qnet using target Q
                    _ = sess.run(Qnet.updateModel,
                                 feed_dict={Qnet.input:np.vstack([exp[0].formatState() for exp in trainingBatch]),
                                            Qnet.actions:np.vstack([exp[1] for exp in trainingBatch]),
                                            Qnet.target:targetQ})
    return None