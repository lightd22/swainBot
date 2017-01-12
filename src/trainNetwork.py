import numpy as np
import tensorflow as tf
import random
from cassiopeia import riotapi
from draftstate import DraftState

import championinfo as cinfo
import matchProcessing as mp
import experienceReplay as er



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

    # For now, we are training off a single draft (my most recent). Later on this will be populated with numEpisode drafts
    matchQueue = mp.buildMatchQueue()
    matchRef = matchQueue.get()
    match = matchRef.match()

    totalSteps = 0
    # Number of steps to take before doing any training. Needs to be at least batchSize to avoid error when sampling from experience replay
    preTrainingSteps = batchSize
    # Number of steps to take between training
    updateFreq = 3 # 3 picks per match -> train after every match

    # Start training
    with tf.Session() as sess:
        # Open saved model (if flagged)
        if loadModel:
            Qnet.saver.restore(sess,"tmp/model.ckpt")
        else:
            # Otherwise, initialize tensorflow variables
            sess.run(Qnet.init)
            team = DraftState.RED_TEAM if match.red_team.win else DraftState.BLUE_TEAM # For now only learn from winning team
            foos = mp.processMatch(matchRef, team, mode = "ban")
            blankExp = foos[0]
            blankState = blankExp[0] 
            estQ = sess.run(Qnet.outQ,
                            feed_dict={Qnet.input:[blankState.formatState()]})
            print("Starting estimates for Q(s,-) from blank state are..")
            print("a \t \t Q(s,a)")
            print("************************")
            for i in range(estQ.shape[1]):
                print("{} \t \t {}".format(i,estQ[0,i]))

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
                if (totalSteps >= preTrainingSteps) and (totalSteps % updateFreq == 0):
                    trainingBatch = experienceReplay.sample(batchSize)

                    #TODO (Devin): Every reference to trainingBatch involves vstacking each column of the batch before using it.. probably better to just have er.sample() return
                    # a numpy array.
                    
                    # Calculate target Q values for each example:
                    # Normally, targetQ is estimated according to 
                    #   targetQ = r + gamma*max_{a} Q(s',a). 
                    # However, for terminating states (where state.evaluateState() == DS.DRAFT_COMPLETE) the target is computed as
                    #   targetQ = r 
                    # therefore, it isn't necessary to feed these states through the ANN to evalutate targetQ. This should save
                    # time as the network complexity increases.
                    updates = []
                    for exp in trainingBatch:
                        startState,_,reward,finalState = exp
                        if finalState.evaluateState() == DraftState.DRAFT_COMPLETE: # Action selection moves to terminal state
                            updates.append(reward)
                        else:                           
                            # Each row in predictedQ gives estimated Q(s',a) values for each possible action for the input state s'.
                            predictedQ = sess.run(Qnet.outQ,
                                                  feed_dict={Qnet.input:[finalState.formatState()]})

                            # To get max_{a} Q(s',a) values take max along *rows* of predictedQ.
                            maxQ = np.max(predictedQ,axis=1)
                            updates.append(reward + Qnet.discountFactor*maxQ)

                    targetQ = np.array(updates)
                    
                    estQ = sess.run(Qnet.outQ,
                                    feed_dict={Qnet.input:np.vstack([exp[0].formatState() for exp in trainingBatch])})
                    print("Current estimates for Q(s,-) for last-pick episode are..")
                    print("a \t \t Q(s,a)")
                    print("************************")
                    for i in range(estQ.shape[1]):
                        print("{} \t \t {}".format(i,estQ[0,i]))

                    # Update Qnet using target Q
                    _ = sess.run(Qnet.updateModel,
                                 feed_dict={Qnet.input:np.vstack([exp[0].formatState() for exp in trainingBatch]),
                                            Qnet.actions:np.array([exp[1] for exp in trainingBatch]),
                                            Qnet.target:targetQ})
        # Once training is complete, save the updated network
        outPath = Qnet.saver.save(sess,"tmp/model.ckpt")
        print("qNet model is saved in file: {}".format(outPath))
    print("***")
    return None
