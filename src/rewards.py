import numpy as np
from draftstate import DraftState as ds

def getReward(state, match):
    """
    Args: 
        state (DraftState): Present state of the draft to be checked for reward
    Returns:
        reward (int): Integer value representing the reward earned for the draft state.

    getReward takes a draft state and returns the immediate reward for reaching that state. The reward is determined by a simple reward table
        1) state is invalid -> reward = -50
        2) state is valid, complete, and the team we are drafting for won -> reward = 100
        3) state is valid, but either incomplete or our team lost  -> reward = 0
    """
    status = state.evaluateState()
    if (status in ds.invalidStates):
        return -50
    elif (status == ds.DRAFT_COMPLETE and state.team == getWinningTeam(match)):
        return 100
    return 0

def getWinningTeam(match):
    """
    Args:
        match (cassiopeia.type.core.match.Match): Cassiopeia-wrapped match data structure returned from Riot's API
    Returns:
        val (int): Integer representing which team won the match. val = DraftState.RED_TEAM if the red team won or val = DraftState.BLUE_TEAM otherwise.

    getWinningTeam returns the winning team of the input match encoded as an integer according to DraftState.
    
    NOTE: getWinningTeam only implicitly assumes that match is of the type cassiopeia.type.core.match.Match, however any data structure with a boolean field red_team.win will
    successfully return a value. This might be useful for testing toy match examples.
    """
    if match.red_team.win:
        return ds.RED_TEAM
    return ds.BLUE_TEAM
