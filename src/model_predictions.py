import experienceReplay as er
import matchProcessing as mp
import championinfo as cinfo
import draftDbOps as dbo
from draftstate import DraftState
from model import Model

import json
import pandas as pd
import numpy as np
import tensorflow as tf
import sqlite3
import math

#path_to_model = "model_predictions/play_ins_rd2/model_play_ins_rd2"
path_to_model = "tmp/models/model_E100"
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
    print("Looking at drafts by winning team")
print("***")


model = Model(path_to_model)

#with open('worlds_matchids_by_stage.txt','r') as infile:
#    data = json.load(infile)
#match_ids = data["groups"]
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
match_ids = dbo.getGameIdsByTournament(cur,"2017/INTL/WRLDS")
matches = [dbo.getMatchData(cur,match_id) for match_id in match_ids]
conn.close()
if(specific_team):
    matches = [match for match in matches if (match["blue_team"]=="tsm" or match["red_team"]=="tsm")]

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
full_diag = {"top1":0, "topk":0, "l2":[],"k":k}
no_rd1_ban_diag = {"top1":0, "topk":0, "l2":[],"k":k}
no_ban_diag = {"top1":0, "topk":0, "l2":[],"k":k}
model_diagnostics = {"full":full_diag, "no_rd1_ban":no_rd1_ban_diag, "no_bans":no_ban_diag}
for match in matches:
    if(specific_team):
        team = DraftState.RED_TEAM if match["red_team"]==specific_team else DraftState.BLUE_TEAM
    else:
        team = DraftState.RED_TEAM if match["winner"]==1 else DraftState.BLUE_TEAM

    experiences = mp.processMatch(match, team)

    pick_count = 0
    print("")
    print("Match: {:2} {:6} vs {:6} winner: {:2}".format(count, match["blue_team"], match["red_team"], match["winner"]))
    for exp in experiences:
        print(" === ")
        print(" Match {}, Pick {}".format(count, pick_count))
        print(" === ")
        state,act,rew,next_state = exp
        cid,pos = act
        if cid == None:
            continue

        predicted_q_values = model.predict([state])
        predicted_q_values = predicted_q_values[0,:]
        submitted_action_id = state.getAction(*act)

        data = [(a,*state.formatAction(a),predicted_q_values[a]) for a in range(len(predicted_q_values))]
        data = [(a,cinfo.championNameFromId(cid),pos,Q) for (a,cid,pos,Q) in data]
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
        # Norms measuring for all submissions
        if(rank == 0):
            model_diagnostics["full"]["top1"] += 1
        if(rank < k):
            model_diagnostics["full"]["topk"] += 1
        model_diagnostics["full"]["l2"].append(err)

        # Norms excluding round 1 bans
        if(pick_count > 2):
            if(rank == 0):
                model_diagnostics["no_rd1_ban"]["top1"] += 1
            if(rank < k):
                model_diagnostics["no_rd1_ban"]["topk"] += 1
            model_diagnostics["no_rd1_ban"]["l2"].append(err)

        # Norms excluding all bans
        if(pos != -1):
            if(rank == 0):
                model_diagnostics["no_bans"]["top1"] += 1
            if(rank < k):
                model_diagnostics["no_bans"]["topk"] += 1
            model_diagnostics["no_bans"]["l2"].append(err)

        print(" Top predictions:")
        print(df.head()) # Print top 5 choices for network
        df.to_pickle("{}/match{}_pick{}.pkl".format(out_dir,count,pick_count))

        pick_count += 1
    count += 1

print("******************")
print("Norm Information:")
for key in model_diagnostics.keys():
    print(" {}".format(key))
    err_list = model_diagnostics[key]["l2"]
    err = math.sqrt((sum([e**2 for e in err_list])/len(err_list)))
    num_predictions = len(err_list)#10*count
    top1 = model_diagnostics[key]["top1"]
    topk = model_diagnostics[key]["topk"]
    k = model_diagnostics[key]["k"]
    print("  Num_predictions = {}".format(num_predictions))
    print("  top 1: count {} -> acc: {:.4}".format(top1, top1/num_predictions))
    print("  top {}: count {} -> acc: {:.4}".format(k, topk, topk/num_predictions))
    print("  l2 error: {:.4}".format(err))
    print("---")
print("******************")
