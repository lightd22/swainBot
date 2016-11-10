import numpy as np
from cassiopeia import riotapi

# Box is just a vacant class with no initial members. This will be used to hold the championNameFromID dictionary.
class Box:
    pass
__m = Box()
__m.championNameFromID = None

def championNameFromID(championID):
    """
    Args:
        championID (int): Integer ID corresponding to the desired champion name.
    Returns:
        name (string): String name of requested champion. If no such champion can be found, returns NULL

    getChampionNameFromID takes a requested championID number and returns the string name of that champion using a championNameFromID dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    """
    if __m.championNameFromID is None:
       populateChampionDictionary()

    if (championID in __m.championNameFromID):
        return __m.championNameFromID[championID]
    return "ERROR: Champion not found"

def validChampionID(championID):
    """
    Checks to see if championID corresponds to a valid champion id code.
    Returns: True if championID is valid. False otherwise.
    Args:
        championID (int): ID of champion to be verified.
    """
    if __m.championNameFromID is None:
       populateChampionDictionary()

    return ((championID > 0) and (championID <= len(__m.championNameFromID)))

def populateChampionDictionary():
    """
    Args:
        None
    Returns:
        True if succesful, False otherwise
    Populates the module dictionary whose keys are champion IDs and values are strings of the corresponding champion's name.
    """
    riotapi.set_region("NA")
    riotapi.set_api_key("71ab791f-d5fe-45b3-8b3a-0368ce261cbe")
    champions = riotapi.get_champions()
    __m.championNameFromID = {champion.id: champion.name for champion in champions}
    if not __m.championNameFromID:
        return False
    return True