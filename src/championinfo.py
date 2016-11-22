import numpy as np
from cassiopeia import riotapi

# Box is a vacant class with no initial members. This will be used to hold the championId list and championId -> name dictionary.

#TODO (Devin): These members should really just be initialized whenever the storage class is created since the very first thing all of these functions
# do is check if they are none or not..
class Box:
    pass
__m = Box()
__m.championNameFromId = None
__m.validChampionIds = None

def championNameFromId(championId):
    """
    Args:
        championId (int): Integer Id corresponding to the desired champion name.
    Returns:
        name (string): String name of requested champion. If no such champion can be found, returns NULL

    getChampionNameFromId takes a requested championId number and returns the string name of that champion using a championNameFromId dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    """
    if __m.championNameFromId is None:
       populateChampionDictionary()

    if (championId in __m.championNameFromId):
        return __m.championNameFromId[championId]
    return "ERROR: Champion not found"

def validChampionId(championId):
    """
    Checks to see if championId corresponds to a valid champion id code.
    Returns: True if championId is valid. False otherwise.
    Args:
        championId (int): Id of champion to be verified.
    """
    if __m.championNameFromId is None:
       populateChampionDictionary()

    return championId in __m.validChampionIds

def getChampionIds():
    """
    Returns a sorted list of valid champion IDs.
    Args:
        None
    Returns:
        validIds (list(ints)): sorted list of valid champion IDs.
    """
    if __m.validChampionIds is None:
        populateChampionDictionary()

    return __m.validChampionIds[:]

def populateChampionDictionary():
    """
    Args:
        None
    Returns:
        True if succesful, False otherwise
    Populates the module dictionary whose keys are champion Ids and values are strings of the corresponding champion's name.
    """
    riotapi.set_region("NA")
    riotapi.set_api_key("71ab791f-d5fe-45b3-8b3a-0368ce261cbe")
    champions = riotapi.get_champions()
    __m.championNameFromId = {champion.id: champion.name for champion in champions}
    __m.validChampionIds = sorted(__m.championNameFromId.keys())
    if not __m.championNameFromId:
        return False
    return True