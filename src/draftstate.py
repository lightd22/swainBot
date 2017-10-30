import numpy as np
from championinfo import championNameFromId, validChampionId

class InvalidDraftState(Exception):
    pass

def get_active_team(submission_count):
    """
    Gets the active team in the draft based on the number of submissions currently present
    Args:
        submission_count (int): number of submissions currently submitted
    Returns:
        DraftState.BLUE_TEAM if blue is active, else DraftState.RED_TEAM
    """
    blue = DraftState.BLUE_TEAM
    red = DraftState.RED_TEAM
    ACTIVE_TEAM = [blue, red, blue, red, blue, red,
                   blue, red, red, blue, blue, red,
                   red, blue, red, blue,
                   red, blue, red]
    if submission_count > len(ACTIVE_TEAM):
        raise

    return ACTIVE_TEAM[submission_count]
    
class DraftState:
    """
    Args:
        team (int) : indicator for which team we are drafting for (RED_TEAM or BLUE_TEAM)
        champ_ids (list(int)) : list of valid championids which are available for drafting.
        num_positions (int) : number of available positions to draft for. Default is 5 for a standard 5x5 draft.

    DraftState is the class responsible for holding and maintaining the current state of the draft. For a given champion with championid c,
    that champion's state with respect to the draft is at most one of:
        - c is banned from selection.
        - c is selected as part of the opponent's team.
        - c is selected as one of our team's position.

     The state of the draft will be stored as a (numChampions) x (numRoles+2) numPy array. If state(c,k) = 1 then:
        - k = 0 -> champion c is banned from selection.
        - k = 1 -> champion c is selected as part of the enemy team.
        - 2 <= k = num_positions+1 -> champion c is selected as position k-1 in our draft.

    Default draft positions are interpreted as:
        Position 1 -> ADC/Marksman (Primary farm)
        Position 2 -> Middle (Secondary farm)
        Position 3 -> Top (Tertiary farm)
        Position 4 -> Jungle (Farming support)
        Position 5 -> Support (Primary support)
    """
    # State codes
    BAN_AND_SUBMISSION = 101
    DUPLICATE_SUBMISSION = 102
    DUPLICATE_ROLE = 103
    INVALID_SUBMISSION = 104
    TOO_MANY_BANS = 105
    TOO_MANY_PICKS = 106
    invalid_states = [BAN_AND_SUBMISSION, DUPLICATE_ROLE, DUPLICATE_SUBMISSION, INVALID_SUBMISSION,
                      TOO_MANY_BANS, TOO_MANY_PICKS]

    DRAFT_COMPLETE = 1
    BLUE_TEAM = 0
    RED_TEAM = 1
    BAN_PHASE_LENGTHS = [6,4] # Number of bans in each ban phase
    NUM_BANS = sum(BAN_PHASE_LENGTHS) # Total number of bans in draft
    PICK_PHASE_LENGTHS = [6,4] # Number of picks in each pick phase
    NUM_PICKS = sum(PICK_PHASE_LENGTHS) # Total number of picks in draft

    def __init__(self, team, champ_ids, num_positions = 5):
        #TODO (Devin): This should make sure that numChampions >= num_positions
        self.num_champions = len(champ_ids)
        self.num_positions = num_positions
        self.num_actions = (self.num_positions+1)*self.num_champions
        self.state_index_to_champ_id = {i:k for i,k in zip(range(self.num_champions),champ_ids)}
        self.champ_id_to_state_index = {k:i for i,k in zip(range(self.num_champions),champ_ids)}
        self.state = np.zeros((self.num_champions, self.num_positions+2), dtype=bool)
        self.picks = []
        self.bans = []

        #TODO (Devin): This should check for an invalid team passed
        self.team = team

        # The dicts pos_to_pos_index and pos_index_to_pos contain the mapping
        # from position labels to indices to the state matrix and vice versa.
        self.positions = [i-1 for i in range(num_positions+2)]
        self.pos_indices = [1,0]
        self.pos_indices.extend(range(2,num_positions+2))
        self.pos_to_pos_index = dict(zip(self.positions,self.pos_indices))
        self.pos_index_to_pos = dict(zip(self.pos_indices,self.positions))

    def getChampId(self,index):
        """
        getChampId returns the valid champion ID corresponding to the given state index. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping will not be trivial. If index is invalid, returns -1.
        Args:
            index (int): location index in the state array of the desired champion.
        Returns:
            champ_id (int): champion ID corresponding to index (as defined by champ_ids)
        """
        if index not in self.state_index_to_champ_id.keys():
            return -1
        return self.state_index_to_champ_id[index]

    def getStateIndex(self,champ_id):
        """
        getStateIndex returns the state index corresponding to the given champion ID. Since champion IDs are not contiguously defined or even necessarily ordered,
        this mapping is non-trivial. If champ_id is invalid, returns -1.
        Args:
            champ_id (int): id of champion to look up
        Returns
            index (int): state index of corresponding champion id
        """
        if champ_id not in self.champ_id_to_state_index.keys():
            return -1
        return self.champ_id_to_state_index[champ_id]

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
        return self.pos_to_pos_index[position]

    def getPosition(self, pos_index):
        """
        getPosition returns the position label corresponding to the given position index into the state matrix.
        If the position index is invalid, returns False.
        Args:
            pos_index (int): position index to look up
        Returns:
            position (int): position label corresponding to this position index
        """
        if pos_index not in self.pos_indices:
            return False
        return self.pos_index_to_pos[pos_index]

    def format_state(self):
        """
        Format the state so the Q-network can process it.
        Args:
            None
        Returns:
            A copy of self.state
        """
        if(self.evaluateState() in DraftState.invalid_states):
            raise InvalidDraftState("Attempting to format an invalid draft state for network input with code {}".format(self.evaluateState()))

        return self.state

    def format_secondary_inputs(self):
        """
        Produces secondary input information (information about filled positions and draft phase)
        to send to Q-network.
        Args:
            None
        Returns:
            Numpy vector of secondary network inputs
        """
        if(self.evaluateState() in DraftState.invalid_states):
            raise InvalidDraftState("Attempting to format an invalid draft state for network input with code {}".format(self.evaluateState()))

        # First segment of information checks whether each position has been or not filled in the state
        # This is done by looking at columns in the subarray corresponding to positions 1 thru 5
        start = self.getPositionIndex(1)
        end = self.getPositionIndex(5)
        secondary_inputs = np.amax(self.state[:,start:end+1],axis=0)

        # Second segment checks if the phase corresponding to this state is a pick phase
        # This is done by counting the number of bans currently submitted. Note that this assumes
        # that state is currently a valid state. If this is not necessarily the case a check can be made using
        # evaluateState().
        num_bans = len(self.bans)
        num_picks = len(self.picks)
        is_pick_phase = False
        if((num_bans==DraftState.BAN_PHASE_LENGTHS[0] and num_picks < DraftState.PICK_PHASE_LENGTHS[0]) or
           (num_bans==DraftState.NUM_BANS and num_picks < DraftState.NUM_PICKS)):
            is_pick_phase = True
        secondary_inputs = np.append(secondary_inputs, is_pick_phase)
        return secondary_inputs

    def formatAction(self,action):
        """
        Format input action into the corresponding tuple (champ_id, position) which describes the input action.
        Args:
            action (int): Action to be interpreted. Assumed to be generated as output of ANN. action is the index
                          of the flattened 'actionable state' matrix
        Returns:
            (championId, position) (tuple of ints): Tuple of integer values which may be passed as arguments to either
            self.addPick() or self.addBan() depending on the value of position. If position = -1 -> action is a ban otherwise action
            is a pick.

        Note: formatAction() explicitly indexes into 'actionable state' matrix which excludes the portion of the state
        matrix corresponding to opponent team submission. In practice this means that (cid, pos) = formatAction(a) will
        never output pos = 0.
        """
        # 'actionable state' is the sub-state of the state matrix with 'enemy picks' column removed.
        actionable_state = self.state[:,1:]
        if(action not in range(actionable_state.size)):
            raise "Invalid action to formatAction()!"
        (state_index, position_index) = np.unravel_index(action,actionable_state.shape)
        # Action corresponds to a submission that we are allowed to make, ie. a pick or a ban.
        # We can't make submissions to the enemy team, so the indicies corresponding to these actions are removed.
        # position_index needs to be shifted by 1 in order to correctly index into full state array
        position_index += 1
        position = self.getPosition(position_index)
        champ_id = self.getChampId(state_index)
        return (champ_id,position)

    def getAction(self, champion_id, position):
        """
        Given a (championId, position) submission pair. Return the corresponding action index in the flattened state array.
        Args:
            championId (int): Valid id of a champion to be picked/banned.
            position (int): Position of champion to be selected. The value of position determines if championId is interpreted as a pick or ban:
                position = -1 -> champion ban submitted.
                position = 0 -> champion selection submitted by the opposing team.
                0 < position <= num_positions -> champion selection submitted by our team for pos = position
        Returns:
            action (int): Action to be interpreted as index into the flattened state vector. If no such action can be found, returns -1

        Note: getAction() explicitly indexes into 'actionable state' matrix which excludes the portion of the state
        matrix corresponding to opponent team submission. In practice this means that a = formatAction(cid,pos) will
        produce an invalid action for pos = 0.
        """
        state_index = self.getStateIndex(champion_id)
        pos_index = self.getPositionIndex(position)
        if ((state_index==-1) or (pos_index not in range(1,self.state.shape[1]))):
            print("Invalid state index or position out of range!")
            print("cid = {}".format(champion_id))
            print("pos = {}".format(position))
            return -1
        # Convert position index for full state matrix into index for actionable state
        pos_index -= 1
        actionable_state = self.state[:,1:]
        action = np.ravel_multi_index((state_index,pos_index),actionable_state.shape)
        return action

    def updateState(self, champion_id, position):
        """
        Attempt to update the current state of the draft and pick/ban lists with a given championId.
        Returns: True is selection was successful, False otherwise
        Args:
            champion_id (int): Id of champion to add to pick list.
            position (int): Position of champion to be selected. The value of position determines if championId is interpreted as a pick or ban:
                position = -1 -> champion ban submitted.
                position = 0 -> champion selection submitted by the opposing team.
                0 < position <= num_positions -> champion selection submitted by our team for pos = position
        """
        # Special case for NULL ban submitted.
        if (champion_id is None and position == -1):
            # Only append NULL bans to ban list (nothing done to state matrix)
            self.bans.append(champion_id)
            return True

        # Submitted picks of the form (champ_id, pos) correspond with the selection champion = champion_id in position = pos.
        # Bans are given pos = -1 and enemy picks pos = 0. However, this is not how they are stored in the state array.
        # Finally this doesn't match indexing used for state array and action vector indexing (which follow state indexing).
        if((position < -1) or (position > self.num_positions) or (not validChampionId(champion_id))):
            return False

        index = self.champ_id_to_state_index[champion_id]
        pos_index = self.getPositionIndex(position)
        if(position == -1):
            self.bans.append(champion_id)
        else:
            self.picks.append(champion_id)

        self.state[index,pos_index] = True
        return True

    def displayState(self):
        #TODO (Devin): Clean up displayState to make it prettier.
        print("=== Begin Draft State ===")
        print("There are {num_picks} picks and {num_bans} bans completed in this draft. \n".format(num_picks=len(self.picks),num_bans=len(self.bans)))

        print("Banned Champions: {0}".format(list(map(championNameFromId, self.bans))))
        print("Picked Champions: {0}".format(list(map(championNameFromId, self.picks))))
        pos_index = self.getPositionIndex(0)
        enemy_draft_ids = list(map(self.getChampId, list(np.where(self.state[:,pos_index])[0])))
        print("Enemy Draft: {0}".format(list(map(championNameFromId,enemy_draft_ids))))

        print("Ally Draft:")
        for pos_index in range(2,len(self.state[0,:])): # Iterate through each position columns in state
            champ_index = np.where(self.state[:,pos_index])[0] # Find non-zero index
            if not champ_index.size: # No pick is found for this position, create a filler string
                draft_name = "--"
            else:
                draft_name = championNameFromId(self.getChampId(champ_index[0]))
            print("Position {p}: {c}".format(p=pos_index-1,c=draft_name))
        print("=== End Draft State ===")

    def canPick(self, champion_id):
        """
        Check to see if a champion is available to be selected.
        Returns: True if champion is a valid selection, False otherwise.
        Args:
            champion_id (int): Id of champion to check for valid selection.
        """
        return ((champion_id not in self.picks) and championinfo.validChampionId(champion_id))

    def canBan(self, champion_id):
        """
        Check to see if a champion is available to be banned.
        Returns: True if champion is a valid ban, False otherwise.
        Args:
            champion_id (int): Id of champion to check for valid ban.
        """
        return ((champion_id not in self.bans) and championinfo.validChampionId(champion_id))

    def addPick(self, champion_id, position):
        """
        Attempt to add a champion to the selected champion list and update the state.
        Returns: True is selection was successful, False otherwise
        Args:
            champion_id (int): Id of champion to add to pick list.
            position (int): Position of champion to be selected. If position = 0 this is interpreted as a selection submitted by the opposing team.
        """
        if((position < 0) or (position > self.num_positions) or (not validChampionId(champion_id))):
            return False
        self.picks.append(champion_id)
        index = self.getStateIndex(champion_id)
        pos_index = self.getPositionIndex(position)
        self.state[index,pos_index] = True
        return True

    def addBan(self, champion_id):
        """
        Attempt to add a champion to the banned champion list and update the state.
        Returns: True is ban was successful, False otherwise
        Args:
            champion_id (int): Id of champion to add to bans.
        """
        if(not validChampionId(champion_id)):
            return False
        self.bans.append(champion_id)
        index = self.getStateIndex(champion_id)
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
                value = BAN_AND_SUBMISSION -> state has a banned champion selected for draft. This will also appear if a ban is submitted which matches an previously submitted champion.
                value = DUPLICATE_SUBMISSION -> state has a champion drafted which is already part of the opposing team or has already been selected by our team.
                value = DUPLICATE_ROLE -> state has multiple champions selected for a single role
                value = INVALID_SUBMISSION -> state has a submission that was included out of the draft phase order (ex pick during ban phase / ban during pick phase)
        """
        # Check for duplicate submissions appearing in picks or bans
        duplicate_picks = set([cid for cid in self.picks if self.picks.count(cid)>1])
        # Need to remove possible NULL bans as duplicates (since these may be legitimate)
        duplicate_bans = set([cid for cid in self.bans if self.bans.count(cid)>1]).difference(set([None]))
        if(len(duplicate_picks)>0 or len(duplicate_bans)>0):
            return DraftState.DUPLICATE_SUBMISSION

        # Check for submissions appearing in both picks and bans
        if(len(set(self.picks).intersection(set(self.bans)))>0):
            # Invalid state includes an already banned champion
            return DraftState.BAN_AND_SUBMISSION

        # Check for different champions that have been submitted for the same role
        for pos in range(2,self.num_positions+2):
            loc = np.argwhere(self.state[:,pos])
            if(len(loc)>1):
                # Invalid state includes multiple champions intended for the same role.
                return DraftState.DUPLICATE_ROLE

        # Check for out of phase submissions
        num_bans = len(self.bans)
        num_picks = len(self.picks)

        if(num_bans > DraftState.NUM_BANS):
            return DraftState.TOO_MANY_BANS
        if(num_picks > DraftState.NUM_PICKS):
            return DraftState.TOO_MANY_PICKS

        ban_cutoffs = [DraftState.BAN_PHASE_LENGTHS[0], DraftState.NUM_BANS]
        pick_cutoffs = [DraftState.PICK_PHASE_LENGTHS[0], DraftState.NUM_PICKS]
        submission_count = num_bans+num_picks
        if(0<=submission_count<ban_cutoffs[0]):
            if num_picks != 0:
                # Invalid number of picks for ban phase 1
                return DraftState.INVALID_SUBMISSION
        if(ban_cutoffs[0]<=submission_count<(ban_cutoffs[0]+pick_cutoffs[0])):
            if num_bans != ban_cutoffs[0]:
                # Invalid number of bans for pick phase 1
                return DraftState.INVALID_SUBMISSION
        if((ban_cutoffs[0]+pick_cutoffs[0])<=submission_count<(ban_cutoffs[1]+pick_cutoffs[0])):
            if num_picks != pick_cutoffs[0]:
                # Invalid number of picks for ban phase 2
                return DraftState.INVALID_SUBMISSION
        if((ban_cutoffs[1]+pick_cutoffs[0])<=submission_count<=(ban_cutoffs[1]+pick_cutoffs[1])):
            if num_bans != ban_cutoffs[1]:
                # Invalid number of bans for pick phase 2
                return DraftState.INVALID_SUBMISSION

        # State is valid, check if draft is complete
        num_enemy_picks = np.sum(self.state[:,0])
        num_ally_picks = np.sum(self.state[:,2:])
        if(num_ally_picks == self.num_positions and num_enemy_picks == self.num_positions):
            # Draft is valid and complete. Note that it isn't necessary
            # to have the full number of valid bans to register a complete draft. This is
            # because teams can be forced to forefit bans due to disciplinary factor (rare)
            # or they can elect to not submit a ban (very rare).
            return DraftState.DRAFT_COMPLETE

        # Draft is valid, but not complete
        return 0
