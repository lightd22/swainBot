import numpy as np

class DraftState:
    """
    Args:
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
        Position 4 -> Jungle (Secondary support)
        Position 5 -> Support (Primary support)
    """

    def __init__(self, numChampions, numPositions = 5):
        self.numChampions = numChampions
        self.numPositions = numPositions
        self.state = np.zeros((numChampions, numPositions+2), dtype=bool)
        self.picks = []
        self.bans = []

    def displayState(self):
        print("Currently there are {numPicks} picks and {numBans} bans in this draft. \n".format(numPicks=len(self.picks),numBans=len(self.bans)))
        
        print("Banned Champions:")
        print(self.bans)

        print("Enemy Draft:")
        print(np.argwhere(self.state[:,1])+1)

        for posIndex in range(2,len(self.state[0,:])):
            champ = np.argwhere(self.state[:,posIndex])+1
            if champ:
                champid = int(champ)
            else:
                champid = "--"
            print("Position {p}: {c}".format(p=posIndex-1,c=champid))
        print("\n")

    def validChampionID(self, championID):
        """
        Checks to see if championID corresponds to a valid champion id code.
        Returns: True if championID is valid. False otherwise.
        Args:
            championID (int): ID of champion to be verified.
        """
        return ((championID > 0) and (championID <= self.numChampions))

    def canPick(self, championID):
        """
        Check to see if a champion is available to be selected.
        Returns: True if champion is a valid selection, False otherwise.
        Args:
            championID (int): ID of champion to check for valid selection.
        """
        return ((championID not in self.picks) and self.validChampionID(championID))

    def canBan(self, championID):
        """
        Check to see if a champion is available to be banned.
        Returns: True if champion is a valid ban, False otherwise.
        Args:
            championID (int): ID of champion to check for valid ban.
        """
        return ((championID not in self.bans) and self.validChampionID(championID))
    
    def addPick(self, championID, position):
        """
        Attempt to add a champion to the selected champion list and update the state.
        Returns: True is selection was successful, False otherwise
        Args:
            championID (int): ID of champion to add to pick list.
            position (int): Position of champion to be selected. If position = 0 this is interpreted as a selection submitted by the opposing team.
        """
        if(not self.canPick(championID) or (position < 0) or (position > self.numPositions)):
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
        if(not self.canBan(championID)):
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
                value = 1 -> state is valid and complete.
            Invalid codes: 
                value = 101 -> state has a banned champion selected for draft.
                value = 102 -> state has a champion drafted which is already part of the opposing team.
                value = 103 -> state has a champion selected for multiple roles (champion selected more than once).
        """
        for champIndex in range(len(self.state[:,0])):
            loc = np.argwhere(self.state[champIndex,:])
            if(len(loc)>1): # State is invalid, check where problem is
                errIndex = int(loc[0]) # Location of first non-zero index fo
                if(errIndex==0):
                    return 101 # Invalid state includes an already banned champion
                elif(errIndex==1):
                    return 102 # Invalid state includes a champion which was selected for the other team
                else:
                    return 103 # Invalid state includes a champion which has been selected for multiple roles
        # State is valid, check if draft is complete
        numEnemyPicks = np.sum(self.state[:,1])
        numAllyPicks = np.sum(self.state[:,2:])
        if(numAllyPicks == self.numPositions and numEnemyPicks == self.numPositions):
            return 1 # Draft is valid and complete
        # Draft is valid, but not complete
        return 0 

