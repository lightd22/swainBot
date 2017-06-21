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
summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
matchRef = summoner.match_list()[0] # Most recent ranked game
match = matchRef.match()
team = DraftState.RED_TEAM if match.red_team.win else DraftState.BLUE_TEAM # We always win!
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
tournament = "2017/Summer_Season/EU"
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
    (champIndex, pos) = a
    cid = s.getChampId(champIndex)
    print("For the {}th selection:".format(count))
    if pos==-1:
        print("  we banned: {}->{}".format(cid,cinfo.championNameFromId(cid)))
    else:
        print("  we selected: {}->{} for position: {}".format(cid,cinfo.championNameFromId(cid), pos))
    print("  we recieved a reward of {} for this selection".format(r))
    print("")
print(cinfo.championIdFromName("thresh"))
print(cinfo.championNameFromId(128))
state = DraftState(team,validChampIds)
inputSize = len(state.formatState())
outputSize = inputSize
layerSize = (536,536)
learningRate = 0.001
regularizationCoeff = 0.01
print("Qnet input size: {}".format(inputSize))
print("Qnet output size: {}".format(outputSize))
print("Using two layers of size: {}".format(layerSize))
print("Using learning rate: {}".format(learningRate))
print("Using regularization strength: {}".format(regularizationCoeff))
Qnet = qNetwork.Qnetwork(inputSize, outputSize, layerSize, learningRate, regularizationCoeff)
tn.trainNetwork(Qnet,3,10,10,False)

# Now if we want to predict what decisions we should make..
print("buffer length={}".format(len(expReplay.buffer)))
myState,action,_,_ = expReplay.buffer[9]
print("")
print("The state we are predicting from is:")
myState.displayState()

# Print out learner's predicted Q-values for myState after training.
with tf.Session() as sess:
    Qnet.saver.restore(sess,"tmp/model.ckpt")
    print("qNetwork restored")

    inputState = np.vstack([myState.formatState()])
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

print("Closing DB connection..")
conn.close()
###
print("**************************************")
print("MOST OF THE STUFF BELOW THIS WONT WORK")
print("**************************************")
###

(r_ChampId,r_Pos) = myState.formatAction(action[0])
print("The champion our network has chosen was: {}".format(cinfo.championNameFromId(r_ChampId)))
print("The position it recommended was: {}".format(r_Pos))


print("{name} is a level {level} summoner on the NA server.".format(name=summoner.name, level=summoner.level))
champions = riotapi.get_champions()
random_champion = random.choice(champions)
print("He enjoys playing LoL on all different champions, like {name}.".format(name=random_champion.name))

#challenger_league = riotapi.get_challenger()
#best_na = challenger_league[0].summoner
#print("He's much better at writing Python code than he is at LoL. He'll never be as good as {name}.".format(name=best_na.name))

print("")
print("********************************")
print("**  Ending Smart Draft Run!   **")
print("********************************")
