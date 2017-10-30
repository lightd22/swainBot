import random
from copy import deepcopy
import numpy as np
import tensorflow as tf
import random

from draftstate import DraftState
import championinfo as cinfo
import draftDbOps as dbo
from rewards import getReward
import qNetwork

def get_active_team(submission_count):
    # This is lazy.. draft order information should be stored somewhere once
    pick_order = [0,1,0,1,0,1, # First phase bans
                  0,1,1,0,0,1, # First phase picks
                  1,0,1,0, # Second phase bans
                  1,0,0,1] # Second phase picks
    return pick_order[submission_count]

def self_train(sess, explore_prob):
    """
    Runs model currently held in TF Session sess through one self training loop. Returns
    negative memory if model fails to complete draft.
    Args:
        sess (tf.Session()): TF Session used to run model.
        explore_prob (float): Probability that each pick will explore state space by submitting a random action
    Returns:
        experience (s,a,r,s') if network submits illegal action from either side of draft
        None if network completes draft without illegal actions
    """
    valid_champ_ids = cinfo.getChampionIds()
    match = {"winner":None} # Blank match for rewards processing
    # Two states are maintained: one corresponding to the perception of the draft
    # according to each of the teams.
    blue_state = DraftState(DraftState.BLUE_TEAM,valid_champ_ids)
    red_state = DraftState(DraftState.RED_TEAM,valid_champ_ids)
    # Draft dictionary holds states for each perspective
    draft = {0:blue_state,1:red_state}

    online_pred = tf.get_default_graph().get_tensor_by_name("online/prediction:0")
    online_input = tf.get_default_graph().get_tensor_by_name("online/inputs:0")
    online_secondary_input = tf.get_default_graph().get_tensor_by_name("online/secondary_inputs:0")

    submission_count = 0
    while(blue_state.evaluateState() != DraftState.DRAFT_COMPLETE and red_state.evaluateState() != DraftState.DRAFT_COMPLETE):
        active_team = get_active_team(submission_count)
        inactive_team = 0 if active_team else 1
        #print("active {}".format(active_team))
        state = draft[active_team]
        start = deepcopy(state)

        if(random.random() < explore_prob):
            # Explore state space by submitting random action
            pred_act = [random.randint(0,state.num_actions-1)]
        else:
            pred_act = sess.run(online_pred,
                            feed_dict={online_input:[state.format_state()],
                            online_secondary_input:[state.format_secondary_inputs()]})
        action = state.formatAction(pred_act[0])
        #print("cid={} pos={}".format(*action))
        # Update active state
        state.updateState(*action)
        if(state.evaluateState() in DraftState.invalid_states):
            return (start,action,getReward(state,match,action,None),state)
        # Update inactive state, remembering to mask non-bans submitted by opponent
        (cid,pos) = action
        inactive_pos = pos if pos==-1 else 0
        draft[inactive_team].updateState(cid,inactive_pos)
        submission_count += 1

    # Return None if network completes draft without illegal submission
    return None

def dueling_networks(path_to_model):
    valid_champ_ids = cinfo.getChampionIds()
    # Two states are maintained: one corresponding to the perception of the draft
    # according to each of the teams.
    blue_state = DraftState(DraftState.BLUE_TEAM,valid_champ_ids)
    red_state = DraftState(DraftState.RED_TEAM,valid_champ_ids)
    draft = {0:blue_state,1:red_state}
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("{path}.ckpt.meta".format(path=path_to_model))
        saver.restore(sess,"{path}.ckpt".format(path=path_to_model))
        online_out = tf.get_default_graph().get_tensor_by_name("online/outputs:0")
        online_pred = tf.get_default_graph().get_tensor_by_name("online/prediction:0")
        online_input = tf.get_default_graph().get_tensor_by_name("online/inputs:0")
        online_secondary_input = tf.get_default_graph().get_tensor_by_name("online/secondary_inputs:0")

        submission_count = 0
        while(blue_state.evaluateState() != DraftState.DRAFT_COMPLETE and red_state.evaluateState() != DraftState.DRAFT_COMPLETE):
            active_team = get_active_team(submission_count)
            inactive_team = 0 if active_team else 1
            print("active {}".format(active_team))
            state = draft[active_team]
            pred_act = sess.run(online_pred,
                                feed_dict={online_input:[state.format_state()],
                                online_secondary_input:[state.format_secondary_inputs()]})
            cid,pos = state.formatAction(pred_act[0])
            print("cid={} pos={}".format(cid,pos))
            # Update active state
            state.updateState(cid,pos)
            # Update inactive state, remembering to mask non-bans submitted by opponent
            inactive_pos = pos if pos==-1 else 0
            draft[inactive_team].updateState(cid,inactive_pos)
            submission_count += 1

    return draft

if __name__ == "__main__":
    model_name = "models/model_E60"
    path_to_model = "tmp/{}".format(model_name)
    print("Restoring model: {}".format(path_to_model))

    draft = dueling_networks(path_to_model)
    draft[0].displayState()
    draft[1].displayState()
