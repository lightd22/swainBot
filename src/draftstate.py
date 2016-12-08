import numpy as np
from championinfo import championNameFromId, validChampionId

class DraftState:
    """
    Args:
        whichTeam (int) : indicator for which team we are drafting for (RED_TEAM or BLUE_TEAM)
        champIds (list(int)) : list of valid championids which are available for drafting.
        numPositions (int) : number of available positions to draft for. Default is 5 for a standard 5x5 draft.

    DraftState is the class responsible for holding and maintaining the current state of the draft. For a given champion with championid c, 
    that champion's state with respect to the draft is at most one of:
        - c is banned from selection.
        - c is selected as part of the opponent's team.
        - c is selected as one of our team's position.
    
     The state of the draft will be stored as a (numChampions) x (numRoles+2) numPy array. If state(c,k) = 1 then:
        - k = 0 -> champion c is banned from selection.
        - k = 1 -> champion c is selected as part of the enemy team.
        - 2 <= k = numPositions+1 -> champion c is selected as position k-1 in our draft.

    Default draft positions are interpreted as:
        Position 1 -> ADC/Marksman (Primary farm)
        Position 2 -> Middle (Secondary farm)
        Position 3 -> Top (Tertiary farm)
        Position 4 -> Jungle (Farming support)
        Position 5 -> Support (Primary support)
    """
    # State codes
    BAN_SELECTED = 101
    DUPLICATE_SELECTION = 102
    DUPLICATE_ROLE = 103
    invalidStates = [BAN_SELECTED, DUPLICATE_ROLE, DUPLICATE_SELECTION]

    DRAFT_COMPLETE = 1
    #TODO (Devin): For 'BAN' draft mode, 
    DRAFT_MODE = 'BAN'
    BLUE_TEAM = 0
    RED_TEAM = 1
    
    def __init__(self, whichTeam, champIds, numPositions = 5):
        #TODO (Devin): This should make sure that numChampions >= numPositions
        self.numChampions = len(champIds)
        self.numPositions = numPositions
        self.stateIndexToChampId = {i:k for i,k in zip(range(self.numChampions),champIds)}
        self.champIdToStateIndex = {k:i for i,k in zip(range(self.numChampions),champIds)}
        self.state = np.zeros((self.numChampions, self.numPositions+2), dtype=bool)
        self.picks = []
        self.bans = []

        #TODO (Devin): This should check for an invalid team passed
        self.team = whichTeam

    def stateIndexToChampId(self,index):
        """
        stateIndexToChampId returns the valid champion ID corresponding to the given state index. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping will not be trivial. If index is invalid, returns -1.
        Args:
            index (int): location index in the state array of the desired champion.
        Returns:
            champId (int): champion ID corresponding to index (as defined by champIds)
        """
        if index not in self.stateIndexToChampId.keys():
            return -1
        return self.stateIndexToChampId[index]

    def champIdToStateIndex(self,champid):
        """
        champIdTostateIndex returns the state index corresponding to the given champion ID. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping will not be trivial. If champid is invalid, returns -1.
        Args:
            champid (int): id of champion to look up
        Returns
            index (int): state index of corresponding champion id
        """
        if champid not in self.champIdToStateIndex.keys():
            return -1
        return self.champIdToStateIndex[champid]

    def formatState(self):
        """
        Format the state array into a vector so the Q-network can process it.
        Args:
            None
        Returns:
            A copy of self.state reshaped as a numpy vector of length numChampions*(numPositions+2)
        """
        return np.reshape(self.state,self.numChampions*(self.numPositions+2))

    def formatAction(self,action):
        """
        Format input action into the corresponding tuple (champid, position) which indexes the state array.
        Args:
            action (int): Action to be interpreted as an index into the state array.
        Returns: 
            (championId, position) (tuple of ints): Tuple of integer values which may be passed as arguments to either
            self.addPick() or self.addBan() depending on the value of position. If position = -1 -> action is a ban otherwise action
            is a pick.    
        """
        (champId,pos) = np.unravel_index(action,self.state.shape)
        print("champId= {}".format(champId))
        print("pos= {}".format(pos))
        #champId += 1
        #pos -= 1
        #return (champId,pos)
        return (0,0)

    def updateState(self, championId, position):
        """
        Attempt to update the current state of the draft and pick/ban lists with a given championId.
        Returns: True is selection was successful, False otherwise
        Args:
            championId (int): Id of champion to add to pick list.
            position (int): Position of champion to be selected. The value of position determines if championId is interpreted as a pick or ban:
                position = -1 -> champion ban submitted. 
                position = 0 -> champion selection submitted by the opposing team.
                0 < position <= numPositions -> champion selection submitted by our team for pos = position
        """
        #TODO: Currently getRewards() does not work correctly if invalid picks are blocked from selection. This should be fixed later.
        #if(not self.canPick(championId) or (position < -1) or (position > self.numPositions)):
        #    return False

        # Devin: As is, our input formatting of championId & position allows for submitted ally picks
        # of the form (champId, pos) to correspond with the selection champion = championId in position = pos. However, this is *not* how they are stored in the state 
        # array. Furthermore this also forces bans to be given pos = -1 and enemy picks pos = 0. Finally this doesn't match indexing used for state array and action vector indexing
        # (which follow state indexing).

        if((position < -1) or (position > self.numPositions) or (not validChampionId(championId))):
            return False

        if(position == -1):
            self.bans.append(championId)
        else:
            self.picks.append(championId)
        index = self.champIdToStateIndex[championId]
        self.state[index,position+1] = True
        return True

    def displayState(self):
        #TODO (Devin): Clean up displayState to make it prettier.
        print("Currently there are {numPicks} picks and {numBans} bans completed in this draft. \n".format(numPicks=len(self.picks),numBans=len(self.bans)))
        
        print("Banned Champions: {0}".format(list(map(championNameFromId, self.bans))))
        enemyDraftIds = [x+1 for x in np.where(self.state[:,1])] # Convert index locations in state to correct championIds
        print("Enemy Draft: {0}".format(list(map(championNameFromId,enemyDraftIds[0]))))

        print("Our Draft:")
        for posIndex in range(2,len(self.state[0,:])): # Iterate through each position column in state
            champ = np.where(self.state[:,posIndex])[0] # Find non-zero index
            if not champ.size: # No pick is found for this position, create a filler string
                draftName = "--"
            else:
                draftName = championNameFromId(int(champ[0])+1)
            print("Position {p}: {c}".format(p=posIndex-1,c=draftName))
        print("\n")

    def canPick(self, championId):
        """
        Check to see if a champion is available to be selected.
        Returns: True if champion is a valid selection, False otherwise.
        Args:
            championId (int): Id of champion to check for valid selection.
        """
        return ((championId not in self.picks) and championinfo.validChampionId(championId))

    def canBan(self, championId):
        """
        Check to see if a champion is available to be banned.
        Returns: True if champion is a valid ban, False otherwise.
        Args:
            championId (int): Id of champion to check for valid ban.
        """
        return ((championId not in self.bans) and championinfo.validChampionId(championId))
    
    def addPick(self, championId, position):
        """
        Attempt to add a champion to the selected champion list and update the state.
        Returns: True is selection was successful, False otherwise
        Args:
            championId (int): Id of champion to add to pick list.
            position (int): Position of champion to be selected. If position = 0 this is interpreted as a selection submitted by the opposing team.
        """
        #TODO: Currently getRewards() does not work correctly if invalid picks are blocked from selection. This should be fixed later.
        #if(not self.canPick(championId) or (position < 0) or (position > self.numPositions)):
        #    return False
        if((position < 0) or (position > self.numPositions) or (not validChampionId(championId))):
            return False
        self.picks.append(championId)
        self.state[championId-1,position+1] = True
        return True
    
    def addBan(self, championId):
        """
        Attempt to add a champion to the banned champion list and update the state.
        Returns: True is ban was successful, False otherwise
        Args:
            championId (int): Id of champion to add to bans.
        """
        #TODO: Currently getRewards() does not work correctly if invalid bans are blocked from selection. This should be fixed later.
        #if(not self.canBan(championId)):
        #    return False
        if(not validChampionId(championId)):
            return False
        self.bans.append(championId)
        self.state[championId-1,0] = True
        return True

    def evaluateState(self):
        """
        evaluateState checks the current state and determines if the draft as it is currently recorded is valid.
        Returns: value (int) - code indicating validitiy of state
            Valid codes:
                value = 0 -> state is valid but incomplete.
                value = DRAFT_COMPLETE -> state is valid and complete.
            Invalid codes: 
                value = BAN_SELECTED -> state has a banned champion selected for draft.
                value = DUPLICATE_SELECTION -> state has a champion drafted which is already part of the opposing team.
                value = DUPLICATE_ROLE -> state has a champion selected for multiple roles (champion selected more than once).
        """
        for champIndex in range(len(self.state[:,0])):
            loc = np.argwhere(self.state[champIndex,:])
            if(len(loc)>1): # State is invalid, check where problem is
                errIndex = int(loc[0]) # Location of first non-zero index fo
                if(errIndex==0):
                    return DraftState.BAN_SELECTED # Invalid state includes an already banned champion
                elif(errIndex==1):
                    return DraftState.DUPLICATE_SELECTION # Invalid state includes a champion which was selected for the other team
                else:
                    return DraftState.DUPLICATE_ROLE # Invalid state includes a champion which has been selected for multiple roles
        # State is valid, check if draft is complete
        numEnemyPicks = np.sum(self.state[:,1])
        numAllyPicks = np.sum(self.state[:,2:])
        if(numAllyPicks == self.numPositions and numEnemyPicks == self.numPositions):
            return DraftState.DRAFT_COMPLETE # Draft is valid and complete
        if(DraftState.DRAFT_MODE == 'BAN' and len(self.bans) == 6):
            return DraftState.DRAFT_COMPLETE # For ban-only drafts, stop once the bans are registered.

        # Draft is valid, but not complete
        return 0 