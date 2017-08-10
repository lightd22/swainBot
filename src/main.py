import random
import numpy as np
from copy import deepcopy
import json
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
    print("",flush=True)

n_matches = 420
match_data = mp.buildMatchPool(n_matches)
match_pool = match_data["matches"]
# Store training match data in a json file (for use later)
with open('match_pool.txt','w') as outfile:
    json.dump(match_data,outfile)
training_matches = match_pool[:400]
validation_matches = match_pool[400:]

# Network parameters
state = DraftState(team,valid_champ_ids)
input_size = state.formatState().shape
output_size = state.num_actions
filter_size = (16,32,64)
regularization_coeff = 7.5e-5#1.5e-4

# Training parameters
batch_size = 32
buffer_size = 4096
n_epoch = 500
discount_factor = 0.9
learning_rate = 2.0e-4

for i in range(1):
    tf.reset_default_graph()
    Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
    n_epoch = n_epoch*(i+1)
    print("Learning on {} matches for {} epochs. lr {:.4e} reg {:4e}".format(len(training_matches),n_epoch, learning_rate, regularization_coeff),flush=True)
    loss,train_acc = tn.trainNetwork(Qnet,training_matches,validation_matches,n_epoch,batch_size,buffer_size,load_model=False,verbose=True)
    print("Learning complete!")
    print("..final training accuracy: {:.4f}".format(train_acc))
    x = [i+1 for i in range(len(loss))]
    fig = plt.figure()
    plt.plot(x,loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim([0,50])
    fig_name = "tmp/loss_figures/annuled_rate/loss_E{}_run_{}.pdf".format(n_epoch,i+1)
    print("Loss figure saved in:{}".format(fig_name),flush=True)
    fig.savefig(fig_name)

# Look at predicted Q values for states in a randomly drawn match
Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
match = random.sample(training_matches,1)[0]
team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM
experiences = mp.processMatch(match,team)
count = 0
# x labels for q val plots
xticks = []
xtick_locs = []
for a in range(state.num_actions):
    cid,pos = state.formatAction(a)
    if cid not in xticks:
        xticks.append(cid)
        xtick_locs.append(a)
xtick_labels = [cinfo.championNameFromId(cid)[:6] for cid in xticks]

tf.reset_default_graph()
with tf.Session() as sess:
    Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
    Qnet.saver.restore(sess,"tmp/model.ckpt")
    for exp in experiences:
        state,act,rew,next_state = exp
        cid,pos = act
        if cid == None:
            continue
        count += 1
        form_act = state.getAction(cid,pos)
        pred_act, pred_Q = sess.run([Qnet.prediction,Qnet.outQ],feed_dict={Qnet.input:[state.formatState()]})
        pred_act = pred_act[0]
        pred_Q = pred_Q[0,:]
        p_cid,p_pos = state.formatAction(pred_act)
        actual = (cinfo.championNameFromId(cid),pos,pred_Q[form_act])
        pred = (cinfo.championNameFromId(p_cid),p_pos,pred_Q[pred_act])
        print("pred:{}, actual:{}".format(pred,actual))
        fig = plt.figure(figsize=(25,5))
        plt.ylabel('$Q(s,a)$')
        plt.xlabel('$a$')
        plt.xticks(xtick_locs, xtick_labels, rotation=70)
        plt.tick_params(axis='x',which='both',labelsize=6)
        x = np.arange(len(pred_Q))
        plt.bar(x,pred_Q, align='center',alpha=0.8,color='b')
        plt.bar(pred_act, pred_Q[pred_act],align='center',color='r')
        plt.bar(form_act, pred_Q[form_act],align='center',color='g')

        fig_name = "tmp/qval_figs/{}.pdf".format(count)
        fig.savefig(fig_name)

# Now if we want to predict what decisions we should make..
myState,action,_,_ = exp_replay.buffer[0]
print("")
print("The state we are predicting from is:")
myState.displayState()

# Print out learner's predicted Q-values for myState after training.
tf.reset_default_graph()
with tf.Session() as sess:
    Qnet = qNetwork.Qnetwork(input_size, output_size, filter_size, learning_rate, discount_factor, regularization_coeff)
    Qnet.saver.restore(sess,"tmp/model.ckpt")
    print("qNetwork restored")

    input_state = [myState.formatState()]
    action = sess.run(Qnet.prediction,feed_dict={Qnet.input:input_state})
    pred_Q = sess.run(Qnet.outQ,feed_dict={Qnet.input:input_state})
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
