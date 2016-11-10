import numpy as np
from championinfo import championNameFromID, validChampionID

class DraftState:
    """
    Args:
        whichTeam (int) : indicator for which team we are drafting for (RED_TEAM or BLUE_TEAM)
        numChampions (int) : number of champions available in drafting.
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
    BLUE_TEAM = 0
    RED_TEAM = 1

    def __init__(self, whichTeam, numChampions, numPositions = 5):
        #TODO (Devin): This should make sure that numChampions >= numPositions
        self.numChampions = numChampions
        self.numPositions = numPositions
        self.state = np.zeros((numChampions, numPositions+2), dtype=bool)
        self.picks = []
        self.bans = []

        #TODO (Devin): This should check for an invalid team passed
        self.team = whichTeam

    def displayState(self):
        #TODO (Devin): Clean up displayState to make it prettier.
        print("Currently there are {numPicks} picks and {numBans} bans completed in this draft. \n".format(numPicks=len(self.picks),numBans=len(self.bans)))
        
        print("Banned Champions: {0}".format(list(map(championNameFromID, self.bans))))
        enemyDraftIDs = [x+1 for x in np.where(self.state[:,1])] # Convert index locations in state to correct championIDs
        print("Enemy Draft: {0}".format(list(map(championNameFromID,enemyDraftIDs[0]))))

        print("Our Draft:")
        for posIndex in range(2,len(self.state[0,:])): # Iterate through each position column in state
            champ = np.where(self.state[:,posIndex])[0] # Find non-zero index
            if not champ.size: # No pick is found for this position, create a filler string
                draftName = "--"
            else:
                draftName = championNameFromID(int(champ[0])+1)
            print("Position {p}: {c}".format(p=posIndex-1,c=draftName))
        print("\n")

    def canPick(self, championID):
        """
        Check to see if a champion is available to be selected.
        Returns: True if champion is a valid selection, False otherwise.
        Args:
            championID (int): ID of champion to check for valid selection.
        """
        return ((championID not in self.picks) and championinfo.validChampionID(championID))

    def canBan(self, championID):
        """
        Check to see if a champion is available to be banned.
        Returns: True if champion is a valid ban, False otherwise.
        Args:
            championID (int): ID of champion to check for valid ban.
        """
        return ((championID not in self.bans) and championinfo.validChampionID(championID))
    
    def addPick(self, championID, position):
        """
        Attempt to add a champion to the selected champion list and update the state.
        Returns: True is selection was successful, False otherwise
        Args:
            championID (int): ID of champion to add to pick list.
            position (int): Position of champion to be selected. If position = 0 this is interpreted as a selection submitted by the opposing team.
        """
        #TODO: Currently getRewards() does not work correctly if invalid picks are blocked from selection. This should be fixed later.
        #if(not self.canPick(championID) or (position < 0) or (position > self.numPositions)):
        #    return False
        if((position < 0) or (position > self.numPositions) or (not validChampionID(championID))):
            return False
        self.picks.append(championID)
        self.state[championID-1,position+1] = True
        return True
    
    def addBan(self, championID):
        """
        Attempt to add a champion to the banned champion list and update the state.
        Returns: True is ban was successful, False otherwise
        Args:
            championID (int): ID of champion to add to bans.
        """
        #TODO: Currently getRewards() does not work correctly if invalid bans are blocked from selection. This should be fixed later.
        #if(not self.canBan(championID)):
        #    return False
        if(not validChampionID(championID)):
            return False
        self.bans.append(championID)
        self.state[championID-1,0] = True
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
        # Draft is valid, but not complete
        return 0 