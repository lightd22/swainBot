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

from optimizeLearningRate import optimizeLearningRate
import matplotlib.pyplot as plt

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

n_epoch = 5
batch_size = 15
buffer_size = 30
n_matches = 100
match_pool = mp.buildMatchPool(n_matches)
training_matches = match_pool[:75]
validation_matches = match_pool[75:]
max_runs = 1
lr_bounds = [-3.5, -2.5]
reg_bounds = [-4., -2.]
discount_factor = 0.5
#print("Beginning learning_rate/regularization optimization..")
#print("max_runs:{}, n_epoch:{}, n_matches:{}, b:{}, B:{}".format(max_runs,n_epoch,n_matches,batch_size,buffer_size))
#optimizeLearningRate(max_runs, n_epoch, training_matches, validation_matches, lr_bounds, reg_bounds,
#                         input_size, output_size, filter_size, discount_factor, buffer_size, batch_size, save = False)


input_size = state.formatState().shape
output_size = state.num_actions
filter_size = (16,32,64)

n_matches = 12
match_pool = mp.buildMatchPool(n_matches)
training_matches = match_pool[:10]
validation_matches = match_pool[10:]

batch_size = 8
buffer_size = 1024
n_epoch = 1500
spinup_epochs = 0

discount_factor = 0.9
learning_rate = 6.0e-4 #1.2e-3 #2.4e-3
regularization_coeff = 7.5e-5#1.5e-4
for i in range(2):
    print("Learning on {} matches for {} epochs. lr {:.4e} reg {:4e}".format(len(training_matches),n_epoch, learning_rate, regularization_coeff))
    Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
    loss,val_acc = tn.trainNetwork(Qnet,training_matches,validation_matches,n_epoch,batch_size,buffer_size,spinup_epochs,load_model=False,verbose=True)
    print("Learning complete!")
    print("..final training accuracy: {:.4f}".format(val_acc))
    x = [i+1 for i in range(len(loss))]
    fig = plt.figure()
    plt.plot(x,loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([0,10])
    fig_name = "tmp/loss_figures/spinup/{}_epoch/loss_E{}_run_{}.png".format(spinup_epochs,n_epoch,i+1)
    fig.savefig(fig_name)


replay = er.ExperienceBuffer(10*len(training_matches))
for match in training_matches:
    team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM
    experiences = mp.processMatch(match,team)
    replay.store(experiences)
with tf.Session() as sess:
    Qnet.saver.restore(sess,"tmp/model.ckpt")

    for exp in replay.buffer:
        state,act,rew,next_state = exp
        cid,pos = act
        form_act = state.getAction(cid,pos)
        pred_act, pred_Q = sess.run([Qnet.prediction,Qnet.outQ],feed_dict={Qnet.input:[state.formatState()]})
        pred_act = pred_act[0]
        pred_cid,pred_pos = state.formatAction(pred_act)
        if(form_act != pred_act):
            state.displayState()
            print("pred: {} in pos {}, actual: {} in pos {}".format(cinfo.championNameFromId(pred_cid),pred_pos,cinfo.championNameFromId(cid),pos))
            print("pred Q: {:.4f}, actual Q: {:.4f}".format(pred_Q[0,pred_act],pred_Q[0,form_act]))

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
