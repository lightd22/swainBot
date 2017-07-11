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

valid_champ_ids = cinfo.getChampionIds()
print("Number of valid championIds: {}".format(len(valid_champ_ids)))

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
exp_replay = er.ExperienceBuffer(10) # just enough buffer to store the first game's experience
exp_replay.store(experiences)
count = 0
for exp in exp_replay.buffer:
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

state = DraftState(team,valid_champ_ids)
nPos = 7 # Positions 1-5 + ban + enemy selection
input_size = state.formatState().shape
# Output from network won't include selecting for other team
output_size = state.num_actions
filter_size = (8,16)

n_epoch = 20
batch_size = 15
buffer_size = 30
n_matches = 100
match_pool = mp.buildMatchPool(n_matches)
training_matches = match_pool[:75]
validation_matches = match_pool[75:]
max_runs = 100
for count in range(max_runs):
    learning_rate = 10**np.random.uniform(-4.,-2.)#0.005
    regularization_coeff = 10**np.random.uniform(-5.,0.)#0.01
    discount_factor = 0.5
    Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
    loss,val_acc = tn.trainNetwork(Qnet,training_matches,validation_matches,n_epoch,batch_size,buffer_size,False)
    print("{}/{}: val_acc: {:.5f}, learn_rate: {:.3e}, reg_coeff: {:.3e}".format(count+1,max_runs,val_acc,learning_rate,regularization_coeff))

# Now if we want to predict what decisions we should make..
myState,action,_,_ = exp_replay.buffer[0]
print("")
print("The state we are predicting from is:")
myState.displayState()

# Print out learner's predicted Q-values for myState after training.
with tf.Session() as sess:
    Qnet.saver.restore(sess,"tmp/model.ckpt")
    print("qNetwork restored")

    input_state = [myState.formatState()]
    action = sess.run(Qnet.prediction,feed_dict={Qnet.input:input_state})
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:input_state})
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

    for exp in exp_replay.buffer:
        state,a,r,nextState = exp
        print("Predicting from state:")
        state.displayState()
        print("")
        predictedAction = sess.run(Qnet.prediction, feed_dict={Qnet.input:[state.formatState()]})
        (cid,pos) = state.formatAction(predictedAction[0])
        print("Network predicts: {}, {}".format(cinfo.championNameFromId(cid),pos))

    input_state = [state.formatState()]
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:input_state})
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
