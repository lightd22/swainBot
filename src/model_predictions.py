import experience_replay as er
import match_processing as mp
import champion_info as cinfo
import draft_db_ops as dbo
from draftstate import DraftState
from model import Model

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import sqlite3
import math

#path_to_model = "model_predictions/spring_2018/week_3/run_2/model_E10"
#path_to_model = "tmp/models/model_E10"
path_to_model = "tmp/model_E{}".format(30)
print("***")
print("Loading Model From: {}".format(path_to_model))
print("***")

out_dir = "model_predictions/dump"
print("***")
print("Outputting To: {}".format(out_dir))
print("***")

specific_team = None#"tsm"
print("***")
if(specific_team):
    print("Looking at drafts by team:{}".format(specific_team))
else:
    print("Looking at drafts submitted by winning team")
print("***")


model = Model(path_to_model)

with open('worlds_matchids_by_stage.txt','r') as infile:
    data = json.load(infile)
#match_ids = data["groups"]
#match_ids.extend(data["knockouts"])
#match_ids.extend(data["finals"])
#match_ids.extend(data["play_ins_rd1"])
#match_ids.extend(data["play_ins_rd2"])
with open('match_pool.txt','r') as infile:
    data = json.load(infile)
match_ids = data['validation_ids']
#match_ids.extend(data['training_ids'])
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
#match_ids = dbo.get_game_ids_by_tournament(cur,"2017/INTL/WRLDS")
matches = [dbo.get_match_data(cur,match_id) for match_id in match_ids]
conn.close()
if(specific_team):
    matches = [match for match in matches if (match["blue_team"]==specific_team or match["red_team"]==specific_team)]

count = 0
print("************************")
print("Match Schedule:")
print("************************")
with open("{}/_match_schedule.txt".format(out_dir),'w') as outfile:
    outfile.write("************************\n")
    for match in matches:
        output_string = "Match {:2}: id: {:5} tourn: {:20} game_no: {:3} {:6} vs {:6} winner: {:2}".format(count, match["id"], match["tournament"], match["tourn_game_id"], match["blue_team"], match["red_team"], match["winner"])
        print(output_string)
        outfile.write(output_string+'\n')
        count += 1
    outfile.write("************************\n")

with open("{}/match_data.json".format(out_dir),'w') as outfile:
    json.dump(matches,outfile)

count = 0
k = 5 # Rank to look for in topk range
full_diag = {"top1":0, "topk":0, "target":0, "l2":[],"k":k}
no_rd1_ban_diag = {"top1":0, "topk":0, "target":0, "l2":[],"k":k}
no_ban_diag = {"top1":0, "topk":0, "target":0, "l2":[],"k":k}
second_phase_only = {"top1":0, "topk":0, "target":0, "l2":[],"k":k}
bans_only = {"top1":0, "topk":0, "target":0, "l2":[],"k":k}
model_diagnostics = {"full":full_diag, "no_rd1_ban":no_rd1_ban_diag, "no_bans":no_ban_diag, "phase_2_only":second_phase_only, "bans":bans_only}
position_distributions = {"phase_1":[0,0,0,0,0], "phase_2":[0,0,0,0,0]}
actual_pos_distributions = {"phase_1":[0,0,0,0,0], "phase_2":[0,0,0,0,0]}
augmentable_picks = {DraftState.BLUE_TEAM:[0,1,4,6,8], DraftState.RED_TEAM:[0,1,3,6]}
targets = [10,10,10,9,8,7,6,6,6,5]
for match in matches:
#    if(specific_team):
#        team = DraftState.RED_TEAM if match["red_team"]==specific_team else DraftState.BLUE_TEAM
#    else:
#        team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM
#    teams = [DraftState.BLUE_TEAM, DraftState.RED_TEAM]
    teams = [DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM]
    for team in teams:

        experiences = mp.process_match(match, team, augment_data=False)

        print("")
        print("Match: {:2} {:6} vs {:6} winner: {:2}".format(count, match["blue_team"], match["red_team"], match["winner"]))
        for pick_count, exp in enumerate(experiences):
            print(" === ")
            print(" Match {}, Pick {}".format(count, pick_count))
            print(" === ")
            state,act,rew,next_state = exp
            cid,pos = act
            if cid == None:
                continue

            predicted_q_values = model.predict([state])
            predicted_q_values = predicted_q_values[0,:]
            submitted_action_id = state.get_action(*act)

            data = [(a,*state.format_action(a),predicted_q_values[a]) for a in range(len(predicted_q_values))]
            data = [(a,cinfo.champion_name_from_id(cid),pos,Q) for (a,cid,pos,Q) in data]
            df = pd.DataFrame(data, columns=['act_id','cname','pos','Q(s,a)'])

            df.sort_values('Q(s,a)',ascending=False,inplace=True)
            df.reset_index(drop=True,inplace=True)

            df['rank'] = df.index
            df['error'] = abs(df['Q(s,a)'][0] - df['Q(s,a)'])/abs(df['Q(s,a)'][0])

            submitted_row = df[df['act_id']==submitted_action_id]
            print(" Submitted action:")
            print(submitted_row)

            rank = submitted_row['rank'].iloc[0]
            err = submitted_row['error'].iloc[0]

            # For picks submitted back-to-back look ahead to next action to see if it was possibly recommended
            if (rank >= k and pick_count in augmentable_picks[team]):#if False:
                _,next_action,_,_ = experiences[pick_count+1]
                cid,_ = next_action
                if(cid):
                    next_action_id = state.get_action(*next_action)
                    next_row = df[df['act_id']==next_action_id]
                    next_rank = next_row['rank'].iloc[0]
                    if(next_rank < k):
                        result = state.update(*next_action)
                        new_exp = (state, act, rew, None)
                        experiences[pick_count+1] = new_exp
                        rank = next_rank
                        print(" AUGMENTED ACTION:")
                        print(next_row)

            t = targets[pick_count]
            # Norms measuring all submissions
            if(rank == 0):
                model_diagnostics["full"]["top1"] += 1
            if(rank < t):
                model_diagnostics["full"]["target"] += 1
            if(rank < k):
                model_diagnostics["full"]["topk"] += 1
            model_diagnostics["full"]["l2"].append(err)

            # Norms excluding round 1 bans
            if(pick_count > 2):
                if(rank == 0):
                    model_diagnostics["no_rd1_ban"]["top1"] += 1
                if(rank < t):
                    model_diagnostics["no_rd1_ban"]["target"] += 1
                if(rank < k):
                    model_diagnostics["no_rd1_ban"]["topk"] += 1
                model_diagnostics["no_rd1_ban"]["l2"].append(err)

            # Norms excluding round 1 completely
            if(pick_count > 5):
                if(rank == 0):
                    model_diagnostics["phase_2_only"]["top1"] += 1
                if(rank < t):
                    model_diagnostics["phase_2_only"]["target"] += 1
                if(rank < k):
                    model_diagnostics["phase_2_only"]["topk"] += 1
                model_diagnostics["phase_2_only"]["l2"].append(err)

            # Norms excluding all bans
            if(pos != -1):
                if(rank == 0):
                    model_diagnostics["no_bans"]["top1"] += 1
                if(rank < t):
                    model_diagnostics["no_bans"]["target"] += 1
                if(rank < k):
                    model_diagnostics["no_bans"]["topk"] += 1
                model_diagnostics["no_bans"]["l2"].append(err)

            # Norms for bans only
            if(pos == -1):
                if(rank == 0):
                    model_diagnostics["bans"]["top1"] += 1
                if(rank < t):
                    model_diagnostics["bans"]["target"] += 1
                if(rank < k):
                    model_diagnostics["bans"]["topk"] += 1
                model_diagnostics["bans"]["l2"].append(err)

            if(rank >= t):
                print(" Top predictions:")
                print(df.head()) # Print top 5 choices for network
            #df.to_pickle("{}/match{}_pick{}.pkl".format(out_dir,count,pick_count))

            # Position distribution for picks
            if(pos > 0):
                top_pos = df.head()["pos"].values.tolist()
                if(pick_count <=5):
                    actual_pos_distributions["phase_1"][pos-1] += 1
                    for pos in top_pos:
                        position_distributions["phase_1"][pos-1] += 1
                else:
                    actual_pos_distributions["phase_2"][pos-1] += 1
                    for pos in top_pos:
                        position_distributions["phase_2"][pos-1] += 1

            pick_count += 1
        count += 1

print("******************")
print("Pick position distributions:")
for phase in ["phase_1", "phase_2"]:
    print("{}: Recommendations".format(phase))
    count = sum(position_distributions[phase])
    for pos in range(len(position_distributions[phase])):
        pos_ratio = position_distributions[phase][pos] / count
        print("  Position {}: Count {:3}, Ratio {:.3}".format(pos+1, position_distributions[phase][pos], pos_ratio))

    print("{}: Actual".format(phase))
    count = sum(actual_pos_distributions[phase])
    for pos in range(len(actual_pos_distributions[phase])):
        pos_ratio = actual_pos_distributions[phase][pos] / count
        print("  Position {}: Count {:3}, Ratio {:.3}".format(pos+1, actual_pos_distributions[phase][pos], pos_ratio))

print("******************")
print("Norm Information:")
for key in sorted(model_diagnostics.keys()):
    print(" {}".format(key))
    err_list = model_diagnostics[key]["l2"]
    err = math.sqrt((sum([e**2 for e in err_list])/len(err_list)))
    num_predictions = len(err_list)
    top1 = model_diagnostics[key]["top1"]
    topk = model_diagnostics[key]["topk"]
    target = model_diagnostics[key]["target"]
    k = model_diagnostics[key]["k"]
    print("  Num_predictions = {}".format(num_predictions))
    print("  top 1: count {} -> acc: {:.4}".format(top1, top1/num_predictions))
    print("  top {}: count {} -> acc: {:.4}".format(k, topk, topk/num_predictions))
    print("  target: count {} -> acc: {:.4}".format(target, target/num_predictions))
    print("  l2 error: {:.4}".format(err))
    print("---")
print("******************")
