import random
import numpy as np
import json
from draftstate import DraftState
import champion_info as cinfo

import experience_replay as er
import match_processing as mp

import qNetwork
from model import Model
import train_network as tn
import tensorflow as tf

import sqlite3

import matplotlib.pyplot as plt
import time


print("")
print("********************************")
print("** Beginning Swain Bot Run! **")
print("********************************")

valid_champ_ids = cinfo.get_champion_ids()
print("Number of valid championIds: {}".format(len(valid_champ_ids)))

with open('worlds_matchids_by_stage.txt','r') as infile:
    worlds_data = json.load(infile)
worlds_play_ins = worlds_data["play_ins_rd1"]
worlds_play_ins.extend(worlds_data["play_ins_rd2"])
worlds_groups = worlds_data["groups"]
worlds_knockouts = worlds_data["knockouts"]
worlds_finals = worlds_data["finals"]

all_worlds_matches = worlds_play_ins+worlds_groups+worlds_knockouts+worlds_finals
random.shuffle(all_worlds_matches)

# Store training match data in a json file (for use later)
reuse_matches = True
if reuse_matches:
    print("Using match data in match_pool.txt.")
    with open('match_pool.txt','r') as infile:
        data = json.load(infile)
    validation_ids = data["validation_ids"]
    training_ids = data["training_ids"]

#    training_ids.extend(worlds_play_ins)

    n_matches = len(validation_ids) + len(training_ids)
    n_training = len(training_ids)
    training_matches = mp.get_matches_by_id(training_ids)
    validation_matches = mp.get_matches_by_id(validation_ids)
    print(n_matches, n_training, len(validation_ids))
else:
    n_val = 3

    match_ids = []
    match_ids.extend(worlds_groups)
    match_ids.extend(worlds_knockouts)
    match_ids.extend(worlds_finals)
    random.shuffle(match_ids)

    validation_ids = match_ids[:n_val]
    training_ids = match_ids[n_val:]

    random.shuffle(validation_ids)
    random.shuffle(training_ids)

    training_matches = mp.get_matches_by_id(training_ids)
    validation_matches = mp.get_matches_by_id(validation_ids)
    with open('match_pool.txt','w') as outfile:
        json.dump({"training_ids":training_ids,"validation_ids":validation_ids},outfile)

print("***")
print("Validation matches:")
count = 0
for match in validation_matches:
    count += 1
    print("Match: {:2} id: {:4} {:6} vs {:6} winner: {:2}".format(count, match["id"], match["blue_team"], match["red_team"], match["winner"]))
    for team in ["blue", "red"]:
        bans = match[team]["bans"]
        picks = match[team]["picks"]
        pretty_bans = []
        pretty_picks = []
        for ban in bans:
            pretty_bans.append(cinfo.champion_name_from_id(ban[0]))
        for pick in picks:
            pretty_picks.append((cinfo.champion_name_from_id(pick[0]), pick[1]))
        print("{} bans:{}".format(team, pretty_bans))
        print("{} picks:{}".format(team, pretty_picks))
    print("")
print("***")

# Network parameters
state = DraftState(DraftState.BLUE_TEAM,valid_champ_ids)
input_size = state.format_state().shape
output_size = state.num_actions
filter_size = (64,128,256)
regularization_coeff = 7.5e-5#1.5e-4
load_model = False

# Training parameters
batch_size = 16#32
buffer_size = 4096#2048
n_epoch = 10
discount_factor = 0.9
learning_rate = 2.0e-5#1.0e-4

for i in range(1):
    tf.reset_default_graph()
    online_net = qNetwork.Qnetwork("online",input_size, output_size, filter_size, learning_rate, regularization_coeff, discount_factor)
    target_net = qNetwork.Qnetwork("target",input_size, output_size, filter_size, learning_rate, regularization_coeff, discount_factor)

    training_ids = []
    training_ids.extend(data["training_ids"])
    #training_ids.extend(mp.build_match_pool(950)["match_ids"])
    training_matches = mp.get_matches_by_id(training_ids)
    print("Learning on {} matches for {} epochs. lr {:.4e} reg {:4e}".format(len(training_matches),n_epoch, learning_rate, regularization_coeff),flush=True)
    loss,train_acc = tn.train_network(online_net,target_net,training_matches,validation_matches,n_epoch,batch_size,buffer_size,dampen_states=False,load_model=load_model,verbose=True)
    print("Learning complete!")
    print("..final training accuracy: {:.4f}".format(train_acc))
    x = [i+1 for i in range(len(loss))]
    fig = plt.figure()
    plt.plot(x,loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim([0,2])
    fig_name = "tmp/loss_figures/annuled_rate/loss_E{}_run_{}.pdf".format(n_epoch,i+1)
    print("Loss figure saved in:{}".format(fig_name),flush=True)
    fig.savefig(fig_name)

# Look at predicted Q values for states in a randomly drawn match
match = random.sample(training_matches,1)[0]
team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM
experiences = mp.process_match(match,team)
count = 0
# x labels for q val plots
xticks = []
xtick_locs = []
for a in range(state.num_actions):
    cid,pos = state.format_action(a)
    if cid not in xticks:
        xticks.append(cid)
        xtick_locs.append(a)
xtick_labels = [cinfo.champion_name_from_id(cid)[:6] for cid in xticks]

tf.reset_default_graph()
path_to_model = "tmp/model_E{}".format(n_epoch)
model = Model(path_to_model)
for exp in experiences:
    state,act,rew,next_state = exp
    cid,pos = act
    if cid == None:
        continue
    count += 1
    form_act = state.get_action(cid,pos)
    pred_act = model.predict_action([state])
    pred_act = pred_act[0]
    pred_Q = model.predict([state])
    pred_Q = pred_Q[0,:]

    p_cid,p_pos = state.format_action(pred_act)
    actual = (cinfo.champion_name_from_id(cid),pos,pred_Q[form_act])
    pred = (cinfo.champion_name_from_id(p_cid),p_pos,pred_Q[pred_act])
    print("pred:{}, actual:{}".format(pred,actual))

    # Plot Q-val figure
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

print("")
print("********************************")
print("**  Ending Smart Draft Run!   **")
print("********************************")
