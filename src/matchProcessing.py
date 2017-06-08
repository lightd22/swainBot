import numpy as np
from cassiopeia import riotapi
import queue
from draftstate import DraftState
from championinfo import getChampionIds, championNameFromId
from rewards import getReward
from copy import deepcopy

def buildMatchQueue(numMatches):
    """
    Args:
        numMatches (int): Number of matches to include in the queue (0 indicates to use the maximum number of matches available)
    Returns:
        matchQueue (Queue of match references): Python Queue structure containing matchIds  to be processed

    This will be responsible for building the queue of matchids that we will use during learning phase.
    *** CURRENTLY ONLY PULLS MY RANKED MATCH HISTORY ***
    """
    #TODO (Devin): This will be responsible for building the queue of matchids that we will use during learning
    # eventually it should recursively look through high mmr player match histories and build up a database of match references.
    matchQueue = queue.Queue(maxsize=numMatches)
    summoner = riotapi.get_summoner_by_name("DOCTOR LIGHT")
    for matchRef in summoner.match_list()[0:numMatches]:
        matchQueue.put(matchRef)
    return matchQueue

def processMatch(matchRef, team, mode):
    """
    processMatch takes an input Cassiopiea match reference and breaks each incremental pick down the draft into experiences (aka "memories").

    Args:
        matchRef (cassiopiea matchReference): Cassiopiea match reference returned by summoner.match_list
        team (DraftState.BLUE_TEAM or DraftState.RED_TEAM): Which team perspective is used to process match
        mode (string): mode = "ban" -> process bans, "draft" -> process champion draft
    Returns:
        experiences ( list(tuple) ): list of experience tuples. Each experience is assumed to be of the form (s, a, r, s') where:
            - s and s' are before and after DraftState states
            - a is stateIndex of selected champion 
            - r is the integer reward obtained from selecting that champion

    NOTE: processMatch() can take **EITHER** side of the draft to parse for memories. This means we can ultimately sample from both winning and losing drafts when training
    using ExperienceBuffer.sample(). 
    
    *** CURRENTLY ONLY PULLS BANS FROM MATCHES BECAUSE CHAMPION PICK ORDER IS UNAVAILABLE IN RIOT'S API ***
    """
    experiences = []
    validChampIds = getChampionIds()
    match = matchRef.match()

    # We can only do ban phases (for now..)
    if mode != "ban":
        print("From matchProcessing.processMatch(): Returning empty experiences list!")
        return experiences

    # Pull pick queue from match reference
    pickQueue = buildPickQueue(match, mode = "ban")

    # Set up draft state
    draft = DraftState(team,validChampIds)

    waitForPick = False
    while not pickQueue.empty():
        # Get next pick from queue
        (currentTeam, nextPick) = pickQueue.get()

        # Memory starts when the next pick belongs to designated team
        if currentTeam == team:
            s = deepcopy(draft)
            # Convert action from championId to stateIndex for storage
            a = draft.champIdToStateIndex[nextPick]
            waitForPick = True

        draft.updateState(nextPick,-1)

        # There are two conditions under which we want to store a memory:
        # 1. Team has made a pick and we're waiting for opposing pick (which has just been submitted)
        # 2. Last pick finishes the draft (eg no further picks in the draft)
        # NOTE: 2. only occurs for red side's final pick.
        if(waitForPick and (currentTeam != team or draft.evaluateState() == DraftState.DRAFT_COMPLETE)):
            r = getReward(draft, match)
            sNext = deepcopy(draft)
            memory = (s, a, r, sNext)
            experiences.append(memory)
            waitForPick = False

    return experiences

def buildPickQueue(match, mode):
    """
    Builds queue of champion picks or bans (depending on mode) in selection order. If mode = 'ban' this produces a queue of tuples
    Args:
        match (cassiopiea match): Cassiopiea match structure to be parsed
        mode (string): mode = 'ban' -> process bans, 'draft' -> process champion draft
    Returns:
        pickQueue (Queue(tuple)): Queue of pick tuples of the form (team, championId). pickQueue is produced in selection order.

    *** CURRENTLY ONLY PRODUCES BAN MODE SELECTIONS SINCE CHAMPION PICK ORDER ISNT AVAILABLE THROUGH RIOT'S API ***
    """
    pickQueue = queue.Queue()

    # Bans are currently made in ABABAB format
    if (mode == "ban"):
        #TODO (Devin): clean this up to be more readable. Maybe interpret actual character strings (eg ABABAB as above) as pick order?
        selectionOrder = [DraftState.BLUE_TEAM, DraftState.RED_TEAM, DraftState.BLUE_TEAM, DraftState.RED_TEAM, DraftState.BLUE_TEAM, DraftState.RED_TEAM]
        redBans = queue.Queue()
        blueBans = queue.Queue()
        for ban in match.red_team.bans:
            redBans.put(ban.champion.id)
        for ban in match.blue_team.bans:
            blueBans.put(ban.champion.id)

        for team in selectionOrder:
            if team == DraftState.BLUE_TEAM:
                pickQueue.put((team,blueBans.get()))
            else:
                pickQueue.put((team,redBans.get()))
    # Champion selections are made in ABBAABBAAB format
    #TODO (Devin): When we include actions corresponding to champion selection, we will have to expand actions to include an encoding of
    # 1. Champion banned (this is handled above)
    # 2. Enemy team pick (champion selection, but no role information)
    # 3. Ally team pick (champion selection + role)
    # This will likely take the form of something like actionId = roleId*numChampion + champId
    # (+1 to avoid issues with championId = 0)

    return pickQueue
