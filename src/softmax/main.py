import champion_info as cinfo
import match_processing as mp
from model import softmax
from trainer import trainer
import json
from draftstate import DraftState
import tensorflow as tf
import experience_replay as er

print("")
print("********************************")
print("** Beginning Swain Bot Run! **")
print("********************************")

valid_champ_ids = cinfo.get_champion_ids()
print("Number of valid championIds: {}".format(len(valid_champ_ids)))

reuse_matches = True
if reuse_matches:
    print("Using match data in match_pool.txt.")
    with open('match_pool.txt','r') as infile:
        data = json.load(infile)
    validation_ids = data["validation_ids"]
    training_ids = data["training_ids"]

    n_matches = len(validation_ids) + len(training_ids)
    n_training = len(training_ids)
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
state = DraftState(DraftState.BLUE_TEAM,valid_champ_ids)
input_size = state.format_state().shape
output_size = state.num_actions
filter_size = (1024,1024)
regularization_coeff = 7.5e-5#1.5e-4

path_to_model = None

# Training parameters
batch_size = 64
n_epoch = 50
learning_rate = 2.0e-5#1.0e-4

with tf.Session() as sess:
    softmax_model = softmax.SoftmaxNetwork("softmax_model", input_size, output_size, filter_size, learning_rate, regularization_coeff)
    sess.run(tf.global_variables_initializer())
    if(path_to_model):
        softmax_model.load(sess, path_to_model)
    trainer = trainer.Trainer(sess, softmax_model, n_epoch, training_matches, validation_matches, batch_size)
    trainer.train()
    softmax_model.save(sess, "/tmp/model.ckpt")
