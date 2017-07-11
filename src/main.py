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

import sqlite3
import draftDbOps as dbo

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
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
tournament = "2017/EU/Summer_Season"
gameIds = dbo.getGameIdsByTournament(cur, tournament)
game = gameIds[0]
match = dbo.getMatchData(cur, game)
team = match["winner"]
bluePicks = match["blue"]["picks"]
blueBans = match["blue"]["bans"]
redPicks = match["red"]["picks"]
redBans = match["red"]["bans"]
print("blue side selections:")
print("   picks:")
for pick in bluePicks:
    print("         {}, position={}".format(cinfo.championNameFromId(pick[0]),pick[1]))
print("    bans:")
for ban in blueBans:
    print("         {}".format(cinfo.championNameFromId(ban[0]),ban[1]))
print("")
print("red side selections:")
for pick in redPicks:
    print("         {}, position={}".format(cinfo.championNameFromId(pick[0]),pick[1]))
print("    bans:")
for ban in redBans:
    print("         {}".format(cinfo.championNameFromId(ban[0]),ban[1]))
print("the winner of this game was: {}".format(match["winner"]))

experiences = mp.processMatch(match,team)
expReplay = er.ExperienceBuffer(10) # just enough buffer to store the first game's experience
expReplay.store(experiences)
count = 0
for exp in expReplay.buffer:
    count+=1
    s,a,r,sNew = exp
    (cid, pos) = a
    print("For the {}th selection:".format(count))
    if pos==-1:
        print("  we banned: {}->{}".format(cid,cinfo.championNameFromId(cid)))
    else:
        print("  we selected: {}->{} for position: {}".format(cid,cinfo.championNameFromId(cid), pos))
    print("  we recieved a reward of {} for this selection".format(r))
    print("")

exp = expReplay.buffer[4]
initial,action,rew,final = exp
(cid,pos) = action
print("attempting to make submission:{}, pos={}".format(cinfo.championNameFromId(cid),pos))
initial.updateState(cid,pos)
print(initial.evaluateState())
initial.displayState()

print("attempting to make submission:{}, pos={}".format(cinfo.championNameFromId(cid),pos))
initial.updateState(cid,pos)
print(initial.evaluateState())
initial.displayState()

state = DraftState(team,validChampIds)
nPos = 7 # Positions 1-5 + ban + enemy selection
inputSize = state.formatState().shape
# Output from network won't include selecting for other team
outputSize = state.numActions
layerSize = (881,536)
learningRate = 0.001
regularizationCoeff = 0.
discountFactor = 0.5
print("Qnet input size: {}".format(inputSize))
print("Qnet output size: {}".format(outputSize))
print("Using two layers of size: {}".format(layerSize))
print("Using learning rate: {}".format(learningRate))
print("Using discountFactor: {}".format(discountFactor))
print("Using regularization strength: {}".format(regularizationCoeff))
Qnet = qNetwork.Qnetwork(inputSize, outputSize, layerSize, learningRate, discountFactor, regularizationCoeff)

# Grab a single match
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
tournament = "2017/EU/Summer_Season"
gameIds = dbo.getGameIdsByTournament(cur, tournament)
game = gameIds[0]
training_match = [dbo.getMatchData(cur, game)]

nEpoch = 500
batchSize = 10
bufferSize = 20*len(training_match)
tn.trainNetwork(Qnet,training_match,nEpoch,batchSize,bufferSize,False)

# Now if we want to predict what decisions we should make..
myState,action,_,_ = expReplay.buffer[0]
print("")
print("The state we are predicting from is:")
myState.displayState()

# Print out learner's predicted Q-values for myState after training.
with tf.Session() as sess:
    Qnet.saver.restore(sess,"tmp/model.ckpt")
    print("qNetwork restored")

    inputState = [myState.formatState()]
    action = sess.run(Qnet.prediction,feed_dict={Qnet.input:inputState})
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:inputState})
    print(pred_Q.shape)
    pred_action = np.argmax(pred_Q, axis=1)
    print("We should be taking action a = {}".format(pred_action[0]))
    print("actionid \t championid \t championName \t position \t qValue")
    print("*****************************************************************")
    for i in range(pred_Q.size):
        (cid,pos) = myState.formatAction(i)
        qVal = pred_Q[0,i]
        print("{} \t \t {} \t \t {:12} \t {} \t \t {:.4f}".format(i, cid, cinfo.championNameFromId(cid),pos,qVal))

    (r_ChampId,r_Pos) = myState.formatAction(action[0])
    print("The champion our network has chosen was: {}".format(cinfo.championNameFromId(r_ChampId)))
    print("The position it recommended was: {}".format(r_Pos))

    for exp in expReplay.buffer:
        state,a,r,nextState = exp
        print("Predicting from state:")
        state.displayState()
        print("")
        predictedAction = sess.run(Qnet.prediction, feed_dict={Qnet.input:[state.formatState()]})
        (cid,pos) = state.formatAction(predictedAction[0])
        print("Network predicts: {}, {}".format(cinfo.championNameFromId(cid),pos))

    inputState = [state.formatState()]
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:inputState})
    print("actionid \t championid \t championName \t position \t qValue")
    print("*****************************************************************")
    for i in range(pred_Q.size):
        (cid,pos) = myState.formatAction(i)
        qVal = pred_Q[0,i]
        print("{} \t \t {} \t \t {:12} \t {} \t \t {:.4f}".format(i, cid, cinfo.championNameFromId(cid),pos,qVal))
    print("We should be taking action a = {}".format(predictedAction[0]))

print("Closing DB connection..")
conn.close()

print("")
print("********************************")
print("**  Ending Smart Draft Run!   **")
print("********************************")
