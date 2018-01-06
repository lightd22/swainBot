import match_processing as mp
from draftstate import DraftState

def validate_matches(match_list):
    """
    Checks if the match data for each element of match_list is valid.
    Args:
        match_list (list(match)): list of match data to validate
    """
    match_count = 0
    for match in match_list:
        match_count += 1
        print("Match {}".format(match_count))
        experiences = mp.process_match(match,DraftState.BLUE_TEAM)
        experiences = mp.process_match(match,DraftState.RED_TEAM)

    return None

if __name__ == "__main__":
    match_data = mp.build_match_pool(1085,randomize=False)
    match_list = match_data["matches"]
    result = validate_matches(match_list)
