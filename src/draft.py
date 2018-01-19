class Draft(object):
    BLUE_TEAM = 0
    RED_TEAM = 1
    BAN = 201
    PICK = 202
    PHASES = [BAN, PICK]

    # Draft specifcations
    # Default specification for drafting structure
    default_draft = [
        (BLUE_TEAM, BAN),
        (RED_TEAM, BAN),
        (BLUE_TEAM, BAN),
        (RED_TEAM, BAN),
        (BLUE_TEAM, BAN),
        (RED_TEAM, BAN),

        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),
        (RED_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),

        (RED_TEAM, BAN),
        (BLUE_TEAM, BAN),
        (RED_TEAM, BAN),
        (BLUE_TEAM, BAN),

        (RED_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),
    ]

    no_bans = [
        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),
        (RED_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),

        (RED_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (BLUE_TEAM, PICK),
        (RED_TEAM, PICK),
    ]
    # Dictionary mapping draft labels to draft structures
    draft_structures = {'default': default_draft,
                        'no_bans': no_bans,
    }

    def __init__(self, draft_type = 'default'):
        self._draft_structure = None
        try:
            self._draft_structure = Draft.draft_structures[draft_type]
        except KeyError:
            print("In draft.py: Draft structure not defined")
            raise

        self.PHASE_LENGTHS = {}
        for phase in Draft.PHASES:
            self.PHASE_LENGTHS[phase] = []

        phase_length = 0
        current_phase = None
        for (team, phase) in self._draft_structure:
            if not current_phase:
                current_phase = phase
            if phase == current_phase:
                phase_length += 1
            else:
                self.PHASE_LENGTHS[current_phase].append(phase_length)
                current_phase = phase
                phase_length = 1
        self.PHASE_LENGTHS[current_phase].append(phase_length) # don't forget last phase
        self.NUM_BANS = sum(self.PHASE_LENGTHS[Draft.BAN]) # Total number of bans in draft
        self.NUM_PICKS = sum(self.PHASE_LENGTHS[Draft.PICK]) # Total number of picks in draft

        # submission_dist[k] gives tuple of counts for pick types just before kth submission is made (last element will hold final submission distribution for draft)
        self.submission_dist = [(0,0,0)]
        for (team, phase) in self._draft_structure:
            (cur_ban, cur_blue, cur_red) = self.submission_dist[-1]
            if phase == Draft.BAN:
                next_dist = (cur_ban+1, cur_blue, cur_red)
            elif team == Draft.BLUE_TEAM:
                next_dist = (cur_ban, cur_blue+1, cur_red)
            else:
                next_dist = (cur_ban, cur_blue, cur_red+1)
            self.submission_dist += [next_dist]

    def get_active_team(self, submission_count):
        """
        Gets the active team in the draft based on the number of submissions currently present
        Args:
            submission_count (int): number of submissions currently submitted to draft
        Returns:
            Draft.BLUE_TEAM if blue is active, else Draft.RED_TEAM
        """
        if submission_count > len(self._draft_structure):
            raise
        elif submission_count == len(self._draft_structure):
            return None
        (team, sub_type) = self._draft_structure[submission_count]
        return team

    def get_active_phase(self, submission_count):
        """
        Returns phase identifier for current phase of the draft based on the number of submissions made.
        Args:
            None
        Returns:
            Draft.BAN if state is in banning phase, otherwise Draft.PICK
        """
        if submission_count > len(self._draft_structure):
            raise
        elif submission_count == len(self._draft_structure):
            return None
        (team, sub_type) = self._draft_structure[submission_count]
        return sub_type

if __name__ == "__main__":
    draft = Draft("default")
    [print(thing) for thing in draft.submission_dist]
