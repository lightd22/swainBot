import random
import numpy as np
from copy import deepcopy
from cassiopeia import riotapi
from draftstate import DraftState
import championinfo as cinfo
from rewards import getReward
import experienceReplay as er
import matchProcessing as mp

class Team(object):
    win = False
    def __init__(self):
        pass

class Match(object):
    red_team = Team()
    def __init__(self,redTeamWon):
        self.red_team.win = redTeamWon

validChampIds = cinfo.getChampionIds()
print("Number of valid championIds: {}".format(len(validChampIds)))

# Simple memory storage loop for this draft.
summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
matchRef = summoner.match_list()[0] # Most recent ranked game
match = matchRef.match()
team = DraftState.RED_TEAM if match.red_team.win else DraftState.BLUE_TEAM # We always win!

experiences = mp.processMatch(matchRef,team,mode="ban")
expReplay = er.ExperienceBuffer(3)
expReplay.store(experiences)
for i in range(3):
    _,a,r,_ = expReplay.buffer[i]
    print("{act} \t {rew}   ".format(act=cinfo.championNameFromId(a), rew=r))

print("{name} is a level {level} summoner on the NA server.".format(name=summoner.name, level=summoner.level))

champions = riotapi.get_champions()
random_champion = random.choice(champions)
print("He enjoys playing LoL on all different champions, like {name}.".format(name=random_champion.name))

challenger_league = riotapi.get_challenger()
best_na = challenger_league[0].summoner
print("He's much better at writing Python code than he is at LoL. He'll never be as good as {name}.".format(name=best_na.name))