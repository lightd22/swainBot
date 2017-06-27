from cassiopeia import riotapi
import queue
from draftstate import DraftState
from championinfo import getChampionIds, championNameFromId
from rewards import getReward
from copy import deepcopy

import sqlite3
import draftDbOps as dbo

import random

import json

def buildMatchQueue(numMatches):
    """
    Args:
        numMatches (int): Number of matches to include in the queue (0 indicates to use the maximum number of matches available)
    Returns:
        matchQueue (Queue of match references): Python Queue structure containing matchIds  to be processed

    This will be responsible for building the queue of matchids that we will use during learning phase.
    """
    #TODO (Devin): This will be responsible for building the queue of matchids that we will use during learning
    # eventually it should recursively look through high mmr player match histories and build up a database of match references.
    matchQueue = queue.Queue(maxsize=numMatches)
    #summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
    #for matchRef in summoner.match_list()[0:numMatches]:
    #    matchQueue.put(matchRef)
    # Pull games from db and convert to match dicts.

    dbName = "competitiveGameData.db"
    conn = sqlite3.connect("tmp/"+dbName)
    cur = conn.cursor()
    tournaments = ["2017/EU/Summer_Season", "2017/NA/Summer_Season", "2017/LCK/Summer_Season",
                    "2017/LPL/Summer_Season", "2017/LMS/Summer_Season"]
    matchPool = []
    for tournament in tournaments:
        gameIds = dbo.getGameIdsByTournament(cur, tournament)
        for game in gameIds:
            match = dbo.getMatchData(cur, game)
            matchPool.append(match)
    print("Number of available matches for training={}".format(len(matchPool)))
    assert numMatches <= len(matchPool), "Not enough matches found to sample!"
    selectedMatches = random.sample(matchPool, numMatches)
    for match in selectedMatches:
        matchQueue.put(match)
    return matchQueue

def processMatch(match, team):
    """
    processMatch takes an input match and breaks each incremental pick and ban down the draft into experiences (aka "memories").

    Args:
        match (dict): match dictionary with pick and ban data for a single game.
        team (DraftState.BLUE_TEAM or DraftState.RED_TEAM): The team perspective that is used to process match
            The selected team has the positions for each pick explicitly included with the experience while the
            "opposing" team has the assigned positions for its champion picks masked.
    Returns:
        experiences ( list(tuple) ): list of experience tuples. Each experience is of the form (s, a, r, s') where:
            - s and s' are DraftState states before and after a single action
            - a is the (stateIndex, position) tuple of selected champion to be banned or picked. position = 0 for submissions
                by the opposing team
            - r is the integer reward obtained from submitting the action a

    processMatch() can take the vantage from both sides of the draft to parse for memories. This means we can ultimately sample from
    both winning drafts (positive reinforcement) and losing drafts (negative reinforcement) when training.
    """
    experiences = []
    validChampIds = getChampionIds()
    # Build queue of actions from match reference
    actionQueue = buildActionQueue(match)

    # Set up draft state
    draft = DraftState(team,validChampIds)

    finishMemory = False
    while not actionQueue.empty():
        # Get next pick from queue
        (submittingTeam, nextPick, position) = actionQueue.get()
        # There are two conditions under which we want to finalize a memory:
        # 1. Non-designated team has finished submitting picks for this phase (ie next submission belongs to the designated team)
        # 2. Draft is complete (no further picks in the draft)
        if submittingTeam == team:
            if finishMemory:
                # This is case 1 to store memory
                r = getReward(draft, match)
                sNext = deepcopy(draft)
                memory = (s, a, r, sNext)
                experiences.append(memory)
                finishMemory = False
            # Memory starts when upcoming pick belongs to designated team
            s = deepcopy(draft)
            # Store action = (champIndex, pos)
            if nextPick is not None:
                # The only time the next pick is None is when a team is forced to
                # forefit a ban due to disciplinary action. We won't allow null bans.
                a = (nextPick, position)
                finishMemory = True
        else:
            # Mask the positions for non-ban selections belonging to the non-designated team
            if position != -1:
                position = 0

        draft.updateState(nextPick, position)

    # Once the queue is empty, store last memory. This is case 2 above.
    # There is always be an outstanding memory at the completion of the draft.
    # RED_TEAM always gets last pick. Therefore:
    #   if team = DraftState.BLUE_TEAM -> There is an outstanding memory from last RED_TEAM submission
    #   if team = DraftState.RED_TEAM -> Memory is open from just before our last submission
    if(draft.evaluateState() == DraftState.DRAFT_COMPLETE):
        assert finishMemory == True
        r = getReward(draft, match)
        sNext = deepcopy(draft)
        memory = (s, a, r, sNext)
        experiences.append(memory)

    return experiences

def buildActionQueue(match):
    """
    Builds queue of champion picks or bans (depending on mode) in selection order. If mode = 'ban' this produces a queue of tuples
    Args:
        match (dict): dictonary structure of match data to be parsed
    Returns:
        actionQueue (Queue(tuple)): Queue of pick tuples of the form (side_id, champion_id, position_id).
            actionQueue is produced in selection order.
    """
    winner = match["winner"]
    actionQueue = queue.Queue()
    phases = {0:{"phaseType":"bans", "pickOrder":["blue", "red", "blue", "red", "blue", "red"]}, # phase 1 bans
              1:{"phaseType":"picks", "pickOrder":["blue", "red", "red", "blue", "blue", "red"]}, # phase 1 picks
              2:{"phaseType":"bans", "pickOrder":["red", "blue", "red", "blue"]}, # phase 2 bans
              3:{"phaseType":"picks","pickOrder":["red", "blue", "blue", "red"]}} # phase 2 picks
    banIndex = 0
    pickIndex = 0
    for phase in range(4):
        phaseType = phases[phase]["phaseType"]
        pickOrder = phases[phase]["pickOrder"]

        numActions = len(pickOrder)
        for pickNum in range(numActions):
            side = pickOrder[pickNum]
            if side == "blue":
                sideId = DraftState.BLUE_TEAM
            else:
                sideId = DraftState.RED_TEAM
            if phaseType == "bans":
                positionId = -1
                index = banIndex
                banIndex += pickNum%2 # Order matters here. index needs to be updated *after* use
            else:
                positionId = match[side][phaseType][pickIndex][1]
                index = pickIndex
                pickIndex += pickNum%2 # Order matters here. index needs to be updated *after* use
            action = (sideId, match[side][phaseType][index][0], positionId)
            actionQueue.put(action)

    return actionQueue
