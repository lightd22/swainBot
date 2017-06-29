import numpy as np
from draftstate import DraftState as ds

def getReward(state, match):
    """
    Args:
        state (DraftState): Present state of the draft to be checked for reward
    Returns:
        reward (int): Integer value representing the reward earned for the draft state.

    getReward takes a draft state and returns the immediate reward for reaching that state. The reward is determined by a simple reward table
        1) state is invalid -> reward = -100
        2) state is valid and the selection was submitted by the winning team -> reward = 100
        3) state is valid, but either incomplete or our team lost  -> reward = 10
    """
    status = state.evaluateState()
    if (status in ds.invalidStates):
        return -50.
    elif (state.team == getWinningTeam(match)):
        return 100.
    return 0.

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
