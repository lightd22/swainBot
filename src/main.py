import random
import numpy as np
from copy import deepcopy
from cassiopeia import riotapi
from draftstate import DraftState
import championinfo as cinfo
from rewards import getReward

class Team(object):
    win = False
    def __init__(self):
        pass

class Match(object):
    red_team = Team()
    def __init__(self,redTeamWon):
        self.red_team.win = redTeamWon

team = DraftState.BLUE_TEAM
draft = DraftState(team, 16)
for i in range(7):
    draft.addBan(i)
for i in range(7,12):
    draft.addPick(i,0)
for i in range(12, 16):
    draft.addPick(i,i-11)

secondDraft = deepcopy(draft)

print("The champion you're looking up is:")
print(cinfo.championNameFromID(76))

draft.displayState()
print("Is this draft done?  ", draft.evaluateState())

draft.addPick(1,5)
draft.displayState()
print("Is this draft done?  ", draft.evaluateState())
match = Match(redTeamWon = True)
print("Our reward for this draft is: {0}".format(getReward(draft,match)))

print()
print("What about our second draft?")
print("Is the second draft done?  ", secondDraft.evaluateState())
secondDraft.displayState()


summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
print("{name} is a level {level} summoner on the NA server.".format(name=summoner.name, level=summoner.level))

champions = riotapi.get_champions()
random_champion = random.choice(champions)
print("He enjoys playing LoL on all different champions, like {name}.".format(name=random_champion.name))

challenger_league = riotapi.get_challenger()
best_na = challenger_league[0].summoner
print("He's much better at writing Python code than he is at LoL. He'll never be as good as {name}.".format(name=best_na.name))