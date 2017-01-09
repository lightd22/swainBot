import random
import numpy as np
from copy import deepcopy
from cassiopeia import riotapi
from draftstate import DraftState
import championinfo as cinfo

import experienceReplay as er
import matchProcessing as mp

import qNetwork
import trainNetwork as tn
import tensorflow as tf

class Team(object):
    win = False
    def __init__(self):
        pass

class Match(object):
    red_team = Team()
    def __init__(self,redTeamWon):
        self.red_team.win = redTeamWon

print("")
print("********************************")
print("** Beginning Smart Draft Run! **")
print("********************************")

validChampIds = cinfo.getChampionIds()
print("Number of valid championIds: {}".format(len(validChampIds)))

# Simple memory storage loop for this draft.
summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
matchRef = summoner.match_list()[0] # Most recent ranked game
match = matchRef.match()
team = DraftState.RED_TEAM if match.red_team.win else DraftState.BLUE_TEAM # We always win!

experiences = mp.processMatch(matchRef,team,mode="ban")
expReplay = er.ExperienceBuffer(3)
expReplay.store(experiences)

for i in range(3):
    _,a,r,_ = expReplay.buffer[i]
    print("{act} \t {rew}   ".format(act=cinfo.championNameFromId(a), rew=r))


# Let's try learning (a lot) from my most recent game..
state = DraftState(DraftState.BLUE_TEAM,validChampIds)
inputSize = len(state.formatState())
outputSize = len(validChampIds)
layerSize = (536,536)
learningRate = 0.001
print("Qnet input size: {}".format(inputSize))
print("Qnet output size: {}".format(outputSize))
print("Using two layers of size: {}".format(layerSize))
print("Using learning rate: {}".format(learningRate))

Qnet = qNetwork.Qnetwork(inputSize, outputSize, layerSize, learningRate)
#tn.trainNetwork(Qnet,1000,30,30)

# Now if we want to predict what bans we should make..
myState,nextBan,_,_ = expReplay.buffer[2]
print("")
print("The state we are predicting from is:")
myState.displayState()

print("")
print("**************")
print("Sanity check:")
blankState,_,_,_ = expReplay.buffer[0]

print("  champion to ban: {}".format(cinfo.championNameFromId(nextBan)))
print("  championId to ban:  {}".format(nextBan))
champIndex = blankState.champIdToStateIndex[nextBan]
print("  state row index of this championId:  {}".format(champIndex))
roleId = 0
act = champIndex*(blankState.numPositions+2)+roleId
print("  action index of banning this champion:  {}".format(act))

blankState.displayState()
blankState.updateState(nextBan,-1)
blankState.displayState()

# Qnet outputs using actionId = championIndex, mp.processMatch() returns (s,a,r,s')
# where a = championId (note the Id, _NOT_ Index), this needs to match indexing of
# Qnet. AKA a = state.champIdToStateIndex(championId). For the time being the roleId
# is unused until the model can predict picks/roles.

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,"tmp/model.ckpt")
    print("qNetwork restored")
    inputState = np.vstack([myState.formatState()])
    action = sess.run(Qnet.prediction,feed_dict={Qnet.input:inputState})
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:inputState})
    pred_action = np.argmax(pred_Q, axis=1)
    print("We should be taking action a = {}".format(pred_action[0]))
    print("actionid \t championid \t championName \t qValue")
    print("********************************************************")
    for i in range(len(validChampIds)):
        (stateid,pos) = myState.formatAction([i])
        cid = myState.stateIndexToChampId[stateid]
        qVal = pred_Q[0,i]
        print("{} \t \t {} \t \t {} \t \t {}".format(i, cid, cinfo.championNameFromId(cid),qVal))

    maxQ = np.max(pred_Q)

(r_ChampId,r_Pos) = state.formatAction(action)
print("The champion our network has chosen was: {}".format(cinfo.championNameFromId(r_ChampId)))
print("The position it recommended was: {}".format(r_Pos))
print("Other champions that are nearby are: {} & {}".format(cinfo.championNameFromId(r_ChampId-1),cinfo.championNameFromId(r_ChampId+1)))


print("{name} is a level {level} summoner on the NA server.".format(name=summoner.name, level=summoner.level))
champions = riotapi.get_champions()
random_champion = random.choice(champions)
print("He enjoys playing LoL on all different champions, like {name}.".format(name=random_champion.name))

#challenger_league = riotapi.get_challenger()
#best_na = challenger_league[0].summoner
#print("He's much better at writing Python code than he is at LoL. He'll never be as good as {name}.".format(name=best_na.name))
