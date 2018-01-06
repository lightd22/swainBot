import numpy as np
from draftstate import DraftState as ds

def getReward(state, match, submitted_action, actual_action):
    """
    Args:
        state (DraftState): Present state of the draft to be checked for reward
        match (dict): record of match which identifies winning team
        submitted_action (tuple(int)): id of action submitted by model
        actual_action (tuple(int)): id of action submitted in observation
    Returns:
        reward (int): Integer value representing the reward earned for the draft state.

    getReward takes a draft state and returns the immediate reward for reaching that state. The reward is determined by a simple reward table
        1) state is invalid -> reward = -10
        2) state is complete, valid, and the selection was submitted by the winning team -> reward = +5
        3) state is complete, valid but the submission was made by the losing team -> reward = +2.5
        3) state is valid, but incomplete  -> reward = 0
    """
    status = state.evaluate()
    if(status in ds.invalid_states):
        return -10.

    reward = 0.
    winner = getWinningTeam(match)
    if(status == ds.DRAFT_COMPLETE and winner is not None):
        if(state.team == winner):
            reward += 5.
        else:
            reward += 2.5

    if(submitted_action == actual_action):
        reward += 1.
    else:
        reward += -0.1

    return reward

def getWinningTeam(match):
    """
    Args:
        match (dict): match dictionary with pick and ban data for a single game.
    Returns:
        val (int): Integer representing which team won the match.
          val = DraftState.RED_TEAM if the red team won
          val = DraftState.BLUE_TEAM if blue team won
          val = None if match does not have data for winning team

    getWinningTeam returns the winning team of the input match encoded as an integer according to DraftState.
    """
    if match["winner"]==0:
        return ds.BLUE_TEAM
    elif match["winner"]==1:
        return ds.RED_TEAM
    return None
