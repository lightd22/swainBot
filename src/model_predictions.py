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
import matplotlib.pyplot as plt
import sqlite3

path_to_model = "tmp/models/model_E{}".format(50)
print("***")
print("Loading Model From: {}".format(path_to_model))
print("***")

out_dir = "model_predictions/play_ins_rd1"
print("***")
print("Outputting To: {}".format(out_dir))
print("***")

model = Model(path_to_model)

with open('match_pool.txt','r') as infile:
    data = json.load(infile)
match_ids = data["validation_ids"]
dbName = "competitiveGameData.db"
conn = sqlite3.connect("tmp/"+dbName)
cur = conn.cursor()
#match_ids = dbo.getGameIdsByTournament(cur,"2017/INTL/WRLDS")
matches = [dbo.getMatchData(cur,match_id) for match_id in match_ids]
conn.close()

count = 0
print("************************")
print("Match Schedule:")
for match in matches:
    print("Match {:2}: id: {:5} tourn: {:20} game_no: {:3} {:6} vs {:6} winner: {:2}".format(count, match["id"], match["tournament"], match["tourn_game_id"], match["blue_team"], match["red_team"], match["winner"]))
    count += 1
print("************************")
with open("{}/_match_schedule.txt".format(out_dir),'w') as outfile:
    outfile.write("************************\n")
    for match in matches:
        outfile.write("Match {:2}: id: {:5} tourn: {:20} game_no: {:3} {:6} vs {:6} winner: {:2}\n".format(count, match["id"], match["tournament"], match["tourn_game_id"], match["blue_team"], match["red_team"], match["winner"]))
        count += 1
    outfile.write("************************\n")

with open("{}/match_data.json".format(out_dir),'w') as outfile:
    json.dump(matches,outfile)

count = 0
for match in matches:
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

        print(" Submitted action:")
        print(df[df['act_id']==submitted_action_id])
        print(" Top predictions:")
        print(df.head()) # Print top 5 choices for network
        df.to_pickle("{}/match{}_pick{}.pkl".format(out_dir,count,pick_count))

        pick_count += 1
    count += 1
