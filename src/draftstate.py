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
    INVALID_SUBMISSION = 104
    invalidStates = [BAN_SELECTED, DUPLICATE_ROLE, DUPLICATE_SELECTION, INVALID_SUBMISSION]

    DRAFT_COMPLETE = 1
    BLUE_TEAM = 0
    RED_TEAM = 1
    BAN_PHASE_LENGTHS = [6,4] # Number of bans in each ban phase
    NUM_BANS = sum(BAN_PHASE_LENGTHS) # Total number of bans in draft
    PICK_PHASE_LENGTHS = [6,4] # Number of picks in each pick phase

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

        # The dicts pos_to_posIndex and posIndex_to_pos contain the mapping
        # from position labels to indices to the state matrix and vice versa.
        self.positions = [i-1 for i in range(numPositions+2)]
        self.posIndices = [1,0]
        self.posIndices.extend(range(2,numPositions+2))
        self.pos_to_posIndex = dict(zip(self.positions,self.posIndices))
        self.posIndex_to_pos = dict(zip(self.posIndices,self.positions))

    def getChampId(self,index):
        """
        getChampId returns the valid champion ID corresponding to the given state index. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping will not be trivial. If index is invalid, returns -1.
        Args:
            index (int): location index in the state array of the desired champion.
        Returns:
            champId (int): champion ID corresponding to index (as defined by champIds)
        """
        if index not in self.stateIndexToChampId.keys():
            return -1
        return self.stateIndexToChampId[index]

    def getStateIndex(self,champid):
        """
        getStateIndex returns the state index corresponding to the given champion ID. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping is non-trivial. If champid is invalid, returns -1.
        Args:
            champid (int): id of champion to look up
        Returns
            index (int): state index of corresponding champion id
        """
        if champid not in self.champIdToStateIndex.keys():
            return -1
        return self.champIdToStateIndex[champid]

    def getPositionIndex(self,position):
        """
        getPositionIndex returns the index of the state matrix corresponding to the given position label.
        If the position is invalid, returns False.
        Args:
            position (int): position label to look up
        Returns:
            index (int): index into the state matrix corresponding to this position
        """
        if position not in self.positions:
            return False
        return self.pos_to_posIndex[position]

    def getPosition(self, posIndex):
        """
        getPosition returns the position label corresponding to the given position index into the state matrix.
        If the position index is invalid, returns False.
        Args:
            posIndex (int): position index to look up
        Returns:
            position (int): position label corresponding to this position index
        """
        if posIndex not in self.posIndices:
            return False
        return self.posIndex_to_pos[posIndex]

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
        Format input action into the corresponding tuple (champid, position) which describes the input action.
        Args:
            action (int): Action to be interpreted. Assumed to be generated as output of ANN. action may be interpreted as the index
                          of the flattened 'actionable state' matrix
        Returns:
            (championId, position) (tuple of ints): Tuple of integer values which may be passed as arguments to either
            self.addPick() or self.addBan() depending on the value of position. If position = -1 -> action is a ban otherwise action
            is a pick.
        """
        # T'actionable state' is the state matrix with 'enemy picks' column removed.
        (stateIndex, positionIndex) = np.unravel_index(action,self.state.shape[:,1:])
        # Action corresponds to a submission that we are allowed to make, ie. a pick or a ban.
        # We can't make submissions to the enemy team, so the indicies corresponding to these actions are removed.
        # positionIndex needs to be shifted by 1 in order to correctly index into full state array
        positionIndex += 1
        position = self.getPosition(positionIndex)
        champId = self.getChampId(stateIndex)
        return (champId,position)

    def getAction(self, championId, position):
        """
        Given a (championId, position) submission pair. Return the corresponding action index in the flattened state array.
        Args:
            championId (int): Valid id of a champion to be picked/banned.
            position (int): Position of champion to be selected. The value of position determines if championId is interpreted as a pick or ban:
                position = -1 -> champion ban submitted.
                position = 0 -> champion selection submitted by the opposing team.
                0 < position <= numPositions -> champion selection submitted by our team for pos = position
        Returns:
            action (int): Action to be interpreted as index into the flattened state vector. If no such action can be found, returns -1
        """
        stateIndex = self.getStateIndex(championId)
        posIndex = self.getPositionIndex(position)
        if ((stateIndex==-1) or (posIndex not in range(self.state.shape[1]))):
            print("Invalid stateIndex or position out of range!")
            print("cid = {}".format(championId))
            print("pos = {}".format(position))
            return -1
        action = np.ravel_multi_index((stateIndex,posIndex),self.state.shape)
        return action

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
        # Special case for NULL ban submitted. This only occurs when a team is penalized to lose ban
        if (championId is None and position == -1):
            # Only append NULL bans to ban list (nothing done to state matrix)
            self.bans.append(championId)
            return True

        # Submitted ally picks of the form (champId, pos) to correspond with the selection champion = championId in position = pos.
        # However, this is not how they are stored in the state array. Bans are given pos = -1 and enemy picks pos = 0.
        # Finally this doesn't match indexing used for state array and action vector indexing (which follow state indexing).
        if((position < -1) or (position > self.numPositions) or (not validChampionId(championId))):
            return False

        index = self.champIdToStateIndex[championId]
        posIndex = self.getPositionIndex(position)
        if(position == -1):
            self.bans.append(championId)
        else:
            self.picks.append(championId)

        self.state[index,posIndex] = True
        return True

    def displayState(self):
        #TODO (Devin): Clean up displayState to make it prettier.
        print("=== Begin Draft State ===")
        print("There are {numPicks} picks and {numBans} bans completed in this draft. \n".format(numPicks=len(self.picks),numBans=len(self.bans)))

        print("Banned Champions: {0}".format(list(map(championNameFromId, self.bans))))
        posIndex = self.getPositionIndex(0)
        enemyDraftIds = list(map(self.getChampId, list(np.where(self.state[:,posIndex])[0])))
        print("Enemy Draft: {0}".format(list(map(championNameFromId,enemyDraftIds))))

        print("Ally Draft:")
        for posIndex in range(2,len(self.state[0,:])): # Iterate through each position columns in state
            champIndex = np.where(self.state[:,posIndex])[0] # Find non-zero index
            if not champIndex.size: # No pick is found for this position, create a filler string
                draftName = "--"
            else:
                draftName = championNameFromId(self.getChampId(champIndex[0]))
            print("Position {p}: {c}".format(p=posIndex-1,c=draftName))
        print("=== End Draft State ===")

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
        index = self.getStateIndex(championId)
        posIndex = self.getPositionIndex(position)
        self.state[index,posIndex] = True
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
        index = self.getStateIndex(chapmionId)
        self.state[index,self.getPositionIndex(-1)] = True
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
                value = INVALID_SUBMISSION -> state has a submission that was included out of the draft phase order (ex pick during ban phase / ban during pick phase)
        """
        # Check for champions that appear multiple times in the state
        for champIndex in range(len(self.state[:,0])):
            loc = np.argwhere(self.state[champIndex,:])
            if(len(loc)>1): # State is invalid, find where problem is
                errIndex = int(loc[0]) # Location of first non-zero index
                if(errIndex==1):
                    # Invalid state includes an already banned champion
                    return DraftState.BAN_SELECTED
                else:
                    # Invalid state either includes a champion which has already been selected by
                    # the other team, or it includes a single champion drafted for multiple roles
                    return DraftState.DUPLICATE_SELECTION
        # Check for different champions that have been submitted for the same role
        for pos in range(2,self.numPositions+2):
            loc = np.argwhere(self.state[:,pos])
            if(len(loc)>1):
                # Invalid state includes multiple champions intended for the same role.
                return DraftState.DUPLICATE_ROLE

        numEnemyPicks = np.sum(self.state[:,0])
        numAllyPicks = np.sum(self.state[:,2:])
        # Check for out of phase submissions
        numBans = len(self.bans)
        numPicks = numEnemyPicks+numAllyPicks
        banCutoffs = [DraftState.BAN_PHASE_LENGTHS[0], DraftState.BAN_PHASE_LENGTHS[0]+DraftState.BAN_PHASE_LENGTHS[1]]
        pickCutoffs = [DraftState.PICK_PHASE_LENGTHS[0], DraftState.PICK_PHASE_LENGTHS[0]+DraftState.PICK_PHASE_LENGTHS[1]]
        # TODO (Devin): This is a litle sloppy but it gets the job done and is fairly
        # understandable. However note that it assumes that ban phase always comes first (reasonable)
        if 0<numBans<banCutoffs[0]:
            if numPicks != 0:
                # Pick submitted during first ban phase
                return DraftState.INVALID_SUBMISSION
        if banCutoffs[0]<numBans<banCutoffs[1]:
            if numPicks != pickCutoffs[0]:
                # Pick submitted during second ban phase
                return DraftState.INVALID_SUBMISSION
        if 0<numPicks<pickCutoffs[0]:
            if numBans != banCutoffs[0]:
                # Ban submitted during first pick phase
                return DraftState.INVALID_SUBMISSION
        if pickCutoffs[0]<numPicks<pickCutoffs[1]:
            if numBans != banCutoffs[1]:
                # Ban submitted during second pick phase
                return DraftState.INVALID_SUBMISSION
        # State is valid, check if draft is complete
        if(numAllyPicks == self.numPositions and numEnemyPicks == self.numPositions):
            # Draft is valid and complete. Note that technically it isn't necessary
            # to have the full number of bans to register a complete draft. This is
            # because teams can be forced to forefit bans due to disciplinary factor (rare)
            # or they can elect to not submit a ban (this hasn't happened)
            return DraftState.DRAFT_COMPLETE

        # Draft is valid, but not complete
        return 0
