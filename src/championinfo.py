import numpy as np
from cassiopeia import riotapi

def getChampionNameFromID(championID):
    """
    Args:
        championID (int): Integer ID corresponding to the desired champion name.
    Returns:
        name (string): String name of requested champion. If no such champion can be found, returns NULL

    getChampionNameFromID takes a requested championID number and returns the string name of that champion using a championNameFromID dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    """
    if(not DraftState.championNameFromID):
        riotapi.set_region("NA")
        riotapi.set_api_key("71ab791f-d5fe-45b3-8b3a-0368ce261cbe")
        champions = riotapi.get_champions()
        championNameFromID = {champion.id: champion.name for champion in champions}

    if (championID in championNameFromID):
        return championNameFromID[championID]
    return ""