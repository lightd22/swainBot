from collections import deque
from draftstate import DraftState
from champion_info import get_champion_ids
from rewards import get_reward
from copy import deepcopy

import sqlite3
import draft_db_ops as dbo

import random
import json

def process_match(match, team, augment_data=True):
    """
    process_match takes an input match and breaks each incremental pick and ban down the draft into experiences (aka "memories").

    Args:
        match (dict): match dictionary with pick and ban data for a single game.
        team (DraftState.BLUE_TEAM or DraftState.RED_TEAM): The team perspective that is used to process match
            The selected team has the positions for each pick explicitly included with the experience while the
            "opposing" team has the assigned positions for its champion picks masked.
        augment_data (optional) (bool): flag controlling the randomized ordering of submissions that do not affect the draft as a whole
    Returns:
        experiences ( list(tuple) ): list of experience tuples. Each experience is of the form (s, a, r, s') where:
            - s and s' are DraftState states before and after a single action
            - a is the (stateIndex, position) tuple of selected champion to be banned or picked. position = 0 for submissions
                by the opposing team
            - r is the integer reward obtained from submitting the action a

    process_match() can take the vantage from both sides of the draft to parse for memories. This means we can ultimately sample from
    both winning drafts (positive reinforcement) and losing drafts (negative reinforcement) when training.
    """
    experiences = []
    valid_champ_ids = get_champion_ids()

    # This section controls data agumentation of the match. Certain submissions in the draft are
    # submitted consecutively by the same team during the same phase (ie team1 pick0 -> team1 pick1).
    # Although these submissions were produced in a particular order, from a draft perspective
    # there is no difference between submissions of the form
    # team1 pick0 -> team1 pick1 vs team1 pick1 -> team0 pickA
    # provided that the two picks are from the same phase (both bans or both picks).
    # Therefore it is possible to augment the order in which these submissions are processed.

    # Note that we can also augment the banning phase if desired. Although these submissions technically
    # fall outside of the conditions listed above, in practice bans made in the same phase are
    # interchangable in order.

    # Build queue of actions from match reference (augmenting if desired)
    augments_list = [
        ("blue","bans",slice(0,3)), # Blue bans 0,1,2 are augmentable
        ("blue","bans",slice(3,5)), # Blue bans 3,4 are augmentable
        ("red","bans",slice(0,3)),
        ("red","bans",slice(3,5)),
        ("blue","picks",slice(1,3)), # Blue picks 1,2 are augmentable
        ("blue","picks",slice(3,5)), # Blue picks 3,4 are augmentable
        ("red","picks",slice(0,2)) # Red picks 0,1 are augmentable
    ]
    if(augment_data):
        augmented_match = deepcopy(match) # Deepcopy match to avoid side effects
        for aug in augments_list:
            (k1,k2,aug_range) = aug
            count = len(augmented_match[k1][k2][aug_range])
            augmented_match[k1][k2][aug_range] = random.sample(augmented_match[k1][k2][aug_range],count)

        action_queue = build_action_queue(augmented_match)
    else:
        action_queue = build_action_queue(match)

    # Set up draft state
    draft = DraftState(team,valid_champ_ids)

    finish_memory = False
    while action_queue:
        # Get next pick from deque
        submission = action_queue.popleft()
        (submitting_team, pick, position) = submission

        # There are two conditions under which we want to finalize a memory:
        # 1. Non-designated team has finished submitting picks for this phase (ie next submission belongs to the designated team)
        # 2. Draft is complete (no further picks in the draft)
        if submitting_team == team:
            if finish_memory:
                # This is case 1 to store memory
                r = get_reward(draft, match, a, a)
                s_next = deepcopy(draft)
                memory = (s, a, r, s_next)
                experiences.append(memory)
                finish_memory = False
            # Memory starts when upcoming pick belongs to designated team
            s = deepcopy(draft)
            # Store action = (champIndex, pos)
            a = (pick, position)
            finish_memory = True
        else:
            # Mask positions for pick submissions belonging to the non-designated team
            if position != -1:
                position = 0

        draft.update(pick, position)

    # Once the queue is empty, store last memory. This is case 2 above.
    # There is always an outstanding memory at the completion of the draft.
    # RED_TEAM always gets last pick. Therefore:
    #   if team = BLUE_TEAM -> There is an outstanding memory from last RED_TEAM submission
    #   if team = RED_TEAM -> Memory is open from just before our last submission
    if(draft.evaluate() == DraftState.DRAFT_COMPLETE):
        assert finish_memory == True
        r = get_reward(draft, match, a, a)
        s_next = deepcopy(draft)
        memory = (s, a, r, s_next)
        experiences.append(memory)
    else:
        print("Week {} match_id {} {} vs {}".format(match["week"], match["id"], match["blue_team"],match["red_team"]))
        draft.display()
        print("Error code {}".format(draft.evaluate()))
        print("Number of experiences {}".format(len(experiences)))
        for experience in experiences:
            _,a,_,_ = experience
            print(a)
        print("")#raise

    return experiences

def build_action_queue(match):
    """
    Builds queue of champion picks or bans (depending on mode) in selection order. If mode = 'ban' this produces a queue of tuples
    Args:
        match (dict): dictonary structure of match data to be parsed
    Returns:
        action_queue (deque(tuple)): deque of pick tuples of the form (side_id, champion_id, position_id).
            action_queue is produced in selection order.
    """
    winner = match["winner"]
    action_queue = deque()
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
            action_queue.append(action)
            completed_actions += 1

    if(completed_actions != 20):
        print("Found a match with missing actions!")
        print("num_actions = {}".format(num_actions))
        print(json.dumps(match, indent=2, sort_keys=True))
    return action_queue

if __name__ == "__main__":
    data = build_match_pool(1, patches=["8.3"])
    matches = data["matches"]
    for match in matches:
        print(match["patch"])
        for team in [DraftState.BLUE_TEAM, DraftState.RED_TEAM]:
            for augment_data in [False, True]:
                experiences = process_match(match, team, augment_data)
                count = 0
                for exp in experiences:
                    _,a,_,_ = exp
                    print("{} - {}".format(count,a))
                    count+=1
                print("")

    data = build_match_pool(0, randomize=False, patches=["8.4","8.5"])
#    matches = data["matches"]
#    for match in matches:
#        print("Week {}, Patch {}: {} vs {}. Winner:{}".format(match["week"], match["patch"], match["blue_team"], match["red_team"], match["winner"]))
