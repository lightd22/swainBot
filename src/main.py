import random
import numpy as np
import pandas
from cassiopeia import riotapi
from draftstate import DraftState

draft = DraftState(16)
for i in range(7):
    draft.addBan(i)
for i in range(7,12):
    draft.addPick(i,0)
for i in range(12, 17):
    draft.addPick(i,i-11)

draft.displayState()
print("Is this draft done?", draft.evaluateState())

riotapi.set_region("NA")
riotapi.set_api_key("71ab791f-d5fe-45b3-8b3a-0368ce261cbe")

summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
print("{name} is a level {level} summoner on the NA server.".format(name=summoner.name, level=summoner.level))

champions = riotapi.get_champions()
random_champion = random.choice(champions)
print("He enjoys playing LoL on all different champions, like {name}.".format(name=random_champion.name))

challenger_league = riotapi.get_challenger()
best_na = challenger_league[0].summoner
print("He's much better at writing Python code than he is at LoL. He'll never be as good as {name}.".format(name=best_na.name))