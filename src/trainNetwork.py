import numpy as np
import tensorflow as tf
import random
from cassiopeia import riotapi
from draftstate import DraftState

import championinfo as cinfo
import matchProcessing as mp
import experienceReplay as er

import sqlite3
import draftDbOps as dbo


def trainNetwork(Qnet, numEpisodes, batchSize, bufferSize, loadModel):
    """
    Args:
        Qnet (qNetwork): Q-network to be trained.
        numEpisodes (int): total number of drafts to be simulated
        batchSize (int): size of each training set sampled from the replay buffer which will be used to update Qnet at a time
        bufferSize (int): size of replay buffer used
        loadModel (bool): flag to reload existing model
    Returns:
        None
    Trains the Q-network Qnet in batches using experience replays.
    """
    print("***")
    print("Beginning training..")
    print("  numEpisodes: {}".format(numEpisodes))
    print("  batchSize: {}".format(batchSize))
    print("  bufferSize: {}".format(bufferSize))

    # Initialize experience replay buffer
    experienceReplay = er.ExperienceBuffer(bufferSize)

    # Train off of list of competitive games from db. To start with we will just train off first game of 2017/EULCS
    matchQueue = mp.buildMatchQueue(1) #(Temporary) right now only learn from first game of the split

    totalSteps = 0
    # Number of steps to take before doing any training. Needs to be at least batchSize to avoid error when sampling from experience replay
    preTrainingSteps = batchSize
    # Number of steps to take between training
    updateFreq = 10 # 5 picks + 5 bans per match -> train after every match (hyperparameter)

    # Start training
    with tf.Session() as sess:
        # Open saved model (if flagged)
        if loadModel:
            Qnet.saver.restore(sess,"tmp/model.ckpt")
        else:
            # Otherwise, initialize tensorflow variables
            sess.run(Qnet.init)

        for episode in range(numEpisodes):
            # Get next match from queue
#            matchRef = matchQueue.get()
            match = matchQueue.get()
            matchQueue.put(match) #(Temporary) put the match we just popped back into queue to check long-time learning

            team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM # For now only learn from winning team
            # Add this match to experience replay
            experiences = mp.processMatch(match, team)
            for experience in experiences:
                experienceReplay.store([experience])
                totalSteps += 1

                # Every updateFreq steps we train the network using the replay buffer
                if (totalSteps >= preTrainingSteps) and (totalSteps % updateFreq == 0):
                    trainingBatch = experienceReplay.sample(batchSize)

                    #TODO (Devin): Every reference to trainingBatch involves vstacking each column of the batch before using it.. probably better to just have er.sample() return
                    # a numpy array.

                    # Calculate target Q values for each example:
                    # For non-temrinal states, targetQ is estimated according to
                    #   targetQ = r + gamma*max_{a} Q(s',a).
                    # For terminating states (where state.evaluateState() == DS.DRAFT_COMPLETE) the target is computed as
                    #   targetQ = r
                    # therefore, it isn't necessary to feed these states through the ANN to evalutate targetQ. This should save
                    # time as the network complexity increases.
                    updates = []
                    for exp in trainingBatch:
                        startState,_,reward,endingState = exp
                        if endingState.evaluateState() == DraftState.DRAFT_COMPLETE: # Action moves to terminal state
                            updates.append(reward)
                        else:
                            # Each row in predictedQ gives estimated Q(s',a) values for each possible action for the input state s'.
                            predictedQ = sess.run(Qnet.outQ,
                                                  feed_dict={Qnet.input:[endingState.formatState()]})

                            # To get max_{a} Q(s',a) values take max along *rows* of predictedQ.
                            maxQ = np.max(predictedQ,axis=1)[0]
                            updates.append(reward + Qnet.discountFactor*maxQ)
                    targetQ = np.array(updates)
                    # Make sure targetQ shape is correct (sometimes np.array likes to return array of shape (batchSize,1))
                    targetQ.shape = (batchSize,)

                    estQ = sess.run(Qnet.outQ,
                                    feed_dict={Qnet.input:np.vstack([exp[0].formatState() for exp in trainingBatch])})
                    print("Current estimates for Q(s,-) for last-pick episode..")
                    print("a \t \t Q(s,a)")
                    print("************************")
                    for i in range(estQ.shape[1]):
                        print("{} \t \t {}".format(i,estQ[0,i]))

                    # Update Qnet using target Q
                    # Experience replay stores action = (champion_id, position) pairs
                    # these need to be converted into the corresponding index of the input vector to the Qnet
                    actions = np.array([startState.getAction(exp[1][0],exp[1][1]) for exp in trainingBatch])
                    _ = sess.run(Qnet.updateModel,
                                 feed_dict={Qnet.input:np.vstack([exp[0].formatState() for exp in trainingBatch]),
                                            Qnet.actions:actions,
                                            Qnet.target:targetQ})
        # Once training is complete, save the updated network
        outPath = Qnet.saver.save(sess,"tmp/model.ckpt")
        print("qNet model is saved in file: {}".format(outPath))
    print("***")
    return None
