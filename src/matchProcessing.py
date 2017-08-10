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

def buildMatchPool(num_matches):
    """
    Args:
        num_matches (int): Number of matches to include in the queue (0 indicates to use the maximum number of matches available)
    Returns:
        match_data (dictionary): dictionary containing two keys:
            "match_ids": list of match_ids for pooled matches
            "matches": list of pooled match data to process

    This will be responsible for building the set of matchids that we will use during learning phase.
    """
    dbName = "competitiveGameData.db"
    conn = sqlite3.connect("tmp/"+dbName)
    cur = conn.cursor()
    tournaments = ["2017/EU/Summer_Season", "2017/NA/Summer_Season", "2017/LCK/Summer_Season",
                    "2017/LPL/Summer_Season", "2017/LMS/Summer_Season", "2017/INTL/MSI"]
    match_pool = []
    # Build list of eligible matche ids
    for tournament in tournaments:
        game_ids = dbo.getGameIdsByTournament(cur, tournament)
        match_pool.extend(game_ids)

    print("Number of available matches for training={}".format(len(match_pool)))
    assert num_matches <= len(match_pool), "Not enough matches found to sample!"
    selected_match_ids = random.sample(match_pool, num_matches)

    selected_matches = []
    for match_id in selected_match_ids:
        match = dbo.getMatchData(cur, match_id)
        selected_matches.append(match)
    conn.close()
    return {"match_ids":selected_match_ids, "matches":selected_matches}

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
    valid_champ_ids = getChampionIds()
    # Build queue of actions from match reference
    action_queue = buildActionQueue(match)

    # Set up draft state
    draft = DraftState(team,valid_champ_ids)

    finish_memory = False
    while not action_queue.empty():
        # Get next pick from queue
        (submitting_team, next_pick, position) = action_queue.get()
        # There are two conditions under which we want to finalize a memory:
        # 1. Non-designated team has finished submitting picks for this phase (ie next submission belongs to the designated team)
        # 2. Draft is complete (no further picks in the draft)
        if submitting_team == team:
            if finish_memory:
                # This is case 1 to store memory
                r = getReward(draft, match)
                s_next = deepcopy(draft)
                memory = (s, a, r, s_next)
                experiences.append(memory)
                finish_memory = False
            # Memory starts when upcoming pick belongs to designated team
            s = deepcopy(draft)
            # Store action = (champIndex, pos)
            a = (next_pick, position)
            finish_memory = True
        else:
            # Mask positions for pick submissions belonging to the non-designated team
            if position != -1:
                position = 0

        draft.updateState(next_pick, position)

    # Once the queue is empty, store last memory. This is case 2 above.
    # There is always be an outstanding memory at the completion of the draft.
    # RED_TEAM always gets last pick. Therefore:
    #   if team = DraftState.BLUE_TEAM -> There is an outstanding memory from last RED_TEAM submission
    #   if team = DraftState.RED_TEAM -> Memory is open from just before our last submission
    if(draft.evaluateState() == DraftState.DRAFT_COMPLETE):
        assert finish_memory == True
        r = getReward(draft, match)
        s_next = deepcopy(draft)
        memory = (s, a, r, s_next)
        experiences.append(memory)
    else:
        print(draft.evaluateState())
        draft.displayState()
        print("{} vs {}".format(match["blue_team"],match["red_team"]))
        print(len(experiences))
        for experience in experiences:
            _,a,_,_ = experience
            print(a)
        raise

    return experiences

def buildActionQueue(match):
    """
    Builds queue of champion picks or bans (depending on mode) in selection order. If mode = 'ban' this produces a queue of tuples
    Args:
        match (dict): dictonary structure of match data to be parsed
    Returns:
        action_queue (Queue(tuple)): Queue of pick tuples of the form (side_id, champion_id, position_id).
            action_queue is produced in selection order.
    """
    winner = match["winner"]
    action_queue = queue.Queue()
    phases = {0:{"phase_type":"bans", "pick_order":["blue", "red", "blue", "red", "blue", "red"]}, # phase 1 bans
              1:{"phase_type":"picks", "pick_order":["blue", "red", "red", "blue", "blue", "red"]}, # phase 1 picks
              2:{"phase_type":"bans", "pick_order":["red", "blue", "red", "blue"]}, # phase 2 bans
              3:{"phase_type":"picks","pick_order":["red", "blue", "blue", "red"]}} # phase 2 picks
    ban_index = 0
    pick_index = 0
    completed_actions = 0
    for phase in range(4):
        phase_type = phases[phase]["phase_type"]
        pick_order = phases[phase]["pick_order"]

        num_actions = len(pick_order)
        for pick_num in range(num_actions):
            side = pick_order[pick_num]
            if side == "blue":
                side_id = DraftState.BLUE_TEAM
            else:
                side_id = DraftState.RED_TEAM
            if phase_type == "bans":
                position_id = -1
                index = ban_index
                ban_index += pick_num%2 # Order matters here. index needs to be updated *after* use
            else:
                position_id = match[side][phase_type][pick_index][1]
                index = pick_index
                pick_index += pick_num%2 # Order matters here. index needs to be updated *after* use
            action = (side_id, match[side][phase_type][index][0], position_id)
            action_queue.put(action)
            completed_actions += 1

    if(completed_actions != 20):
        print("Found a match with missing actions!")
        print("num_actions = {}".format(num_actions))
        print(json.dumps(match, indent=2, sort_keys=True))
    return action_queue
