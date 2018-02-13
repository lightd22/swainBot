import random
import numpy as np
import json
import sqlite3
import matplotlib.pyplot as plt
import time

from draftstate import DraftState
import champion_info as cinfo
import match_processing as mp

from models import qNetwork, softmax
from trainer import DDQNTrainer, SoftmaxTrainer
from models.inference_model import QNetInferenceModel, SoftmaxInferenceModel

import tensorflow as tf

print("")
print("********************************")
print("** Beginning Swain Bot Run! **")
print("********************************")

valid_champ_ids = cinfo.get_champion_ids()
print("Number of valid championIds: {}".format(len(valid_champ_ids)))

# Store training match data in a json file (for reuse later)
reuse_matches = True
val_count = 60
save_match_pool = False

validation_ids = []
training_ids = []
if reuse_matches:
    print("Using match data in match_pool.txt.")
    with open('match_pool.txt','r') as infile:
        data = json.load(infile)
    validation_ids = data["validation_ids"]
    training_ids = data["training_ids"]

val_diff = val_count - len(validation_ids)
if(val_diff > 0):
    print("Insufficient number of validation matches. Attempting to add difference..")
    current = []
    current.extend(validation_ids)
    current.extend(training_ids)

    total = mp.build_match_pool(0, randomize=False)["match_ids"]
    new = list(set(total)-set(current))
    assert(len(new) >= val_diff), "Not enough new matches to match required validation count!"
    random.shuffle(new)
    validation_ids.extend(new[:val_diff])
    training_ids.extend(new[val_diff:])
    save_match_pool = True

if(save_match_pool):
    print("Saving match pool to match_pool.txt..")
    with open('match_pool.txt','w') as outfile:
        json.dump({"training_ids":training_ids,"validation_ids":validation_ids},outfile)

training_matches = mp.get_matches_by_id(training_ids)
validation_matches = mp.get_matches_by_id(validation_ids)

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
state = DraftState(DraftState.BLUE_TEAM, valid_champ_ids)
input_size = state.format_state().shape
output_size = state.num_actions
filter_size = (1024,1024)
regularization_coeff = 7.5e-5#1.5e-4
path_to_model = None#"model_predictions/spring_2018/week_3/model_E{}.ckpt".format(30)#None
load_path = "tmp/ddqn_model.ckpt"

# Training parameters
batch_size = 16#32
buffer_size = 4096#2048
n_epoch = 45
discount_factor = 0.9
learning_rate = 2.0e-5#1.0e-4

time.sleep(2.)
for i in range(1):
    print("Learning on {} matches for {} epochs. lr {:.4e} reg {:4e}".format(len(training_matches),n_epoch, learning_rate, regularization_coeff),flush=True)

    tf.reset_default_graph()
    name = "softmax"
    out_path = "tmp/{}_model_E{}.ckpt".format(name, n_epoch)
    softnet = softmax.SoftmaxNetwork(name, out_path, input_size, output_size, filter_size, learning_rate, regularization_coeff)
    trainer = SoftmaxTrainer(softnet, n_epoch, training_matches, validation_matches, batch_size, load_path=None)
    summaries = trainer.train()

    tf.reset_default_graph()
    name = "ddqn"
    out_path = "tmp/{}_model_E{}.ckpt".format(name, n_epoch)
    ddqn = qNetwork.Qnetwork(name, out_path, input_size, output_size, filter_size, learning_rate, regularization_coeff, discount_factor)
    trainer = DDQNTrainer(ddqn, n_epoch, training_matches, validation_matches, batch_size, buffer_size, load_path)
    summaries = trainer.train()

    print("Learning complete!")
    print("..final training accuracy: {:.4f}".format(summaries["train_acc"][-1]))
    x = [i+1 for i in range(len(summaries["loss"]))]
    fig = plt.figure()
    plt.plot(x,summaries["loss"])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.ylim([0,2])
    fig_name = "tmp/loss_figures/annuled_rate/loss_E{}_run_{}.pdf".format(n_epoch,i+1)
    print("Loss figure saved in:{}".format(fig_name),flush=True)
    fig.savefig(fig_name)

    fig = plt.figure()
    plt.plot(x, summaries["train_acc"], x, summaries["val_acc"])
    fig_name = "tmp/acc_figs/acc_E{}_run_{}.pdf".format(n_epoch,i+1)
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
path_to_model = "tmp/ddqn_model"#"tmp/model_E{}".format(n_epoch)
model = InferenceModel(name="infer", path=path_to_model)
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
print("**  Ending Swain Bot Run!   **")
print("********************************")
