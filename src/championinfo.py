import numpy as np
from cassiopeia import riotapi
import re
from myRiotApiKey import api_key

# Box is a vacant class with no initial members. This will be used to hold the champion_id list and champion_id <-> name dictionaries.

#TODO (Devin): These members should really just be initialized whenever the storage class is created since the very first thing all of these functions
# do is check if they are none or not..
class Box:
    pass
__m = Box()
__m.championNameFromId = None
__m.championIdFromName = None
__m.validChampionIds = None
__m.championAliases = {
"blitz": "blitzcrank",
"gp": "gangplank",
"jarvan": "jarvaniv",
"cait": "caitlyn",
"lb": "leblanc",
"cass": "cassiopeia",
"ori": "orianna",
"lee": "leesin",
"vlad": "vladimir",
"j4": "jarvaniv",
"as": "aurelionsol", # who the fuck thinks this is unique?
"kass": "kassadin",
"tk": "tahmkench",
"malz": "malzahar",
"sej": "sejuani",
"nid": "nidalee",
"aurelion": "aurelionsol",
"mundo": "drmundo",
"tahm": "tahmkench"
}
class AliasException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
        self.message = message

def convertChampionAlias(alias):
    """
    Args:
        alias (string): lowercase and pruned string alias for a champion
    Returns:
        name (string): lowercase and pruned string name for champion

    convertChampionAlias converts a given champion alias (ie "blitz")
    and returns the version of that champions proper name which is suitable for passing to
    championIdFromName(). If no such alias can be found, this raises an AliasException.

    Example: name = convertChampionAlias("blitz") will yield name = "blitzcrank"
    """
    if (alias == "none"):
        return None
    try:
        if (alias in __m.championAliases):
            return __m.championAliases[alias]
        else:
            raise AliasException("Champion alias not found!", alias)
    except AliasException as e:
        print("*****")
        print(e.message)
        print("Offending alias: {}".format(e.errors))
        print("*****")
        raise

def championNameFromId(champion_id):
    """
    Args:
        champion_id (int): Integer Id corresponding to the desired champion name.
    Returns:
        name (string): String name of requested champion. If no such champion can be found, returns NULL

    getChampionNameFromId takes a requested champion_id number and returns the string name of that champion using a championNameFromId dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    """
    if __m.championNameFromId is None:
       populateChampionDictionary()

    if (champion_id in __m.championNameFromId):
        return __m.championNameFromId[champion_id]
    return None

def championIdFromName(champion_name):
    """
    Args:
        champion_name (string): lowercase and pruned string label corresponding to the desired champion id.
    Returns:
        id (int): id of requested champion. If no such champion can be found, returns NULL

    getChampionIdFromName takes a requested champion name and returns the id label of that champion using a championIdFromName dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    Note that champion_name should be all lowercase and have any non-alphanumeric characters (including whitespace) removed.
    """
    if __m.championIdFromName is None:
       populateChampionDictionary()

    if (champion_name in __m.championIdFromName):
        return __m.championIdFromName[champion_name]
    return None

def validChampionId(champion_id):
    """
    Checks to see if champion_id corresponds to a valid champion id code.
    Returns: True if champion_id is valid. False otherwise.
    Args:
        champion_id (int): Id of champion to be verified.
    """
    if __m.championNameFromId is None:
       populateChampionDictionary()

    return champion_id in __m.validChampionIds

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
    riotapi.set_api_key(api_key)
    champions = riotapi.get_champions()
    __m.championNameFromId = {champion.id: champion.name for champion in champions}
    __m.championIdFromName = {re.sub("[^A-Za-z0-9]+", "", champion.name.lower()): champion.id for champion in champions}
    __m.validChampionIds = sorted(__m.championNameFromId.keys())
    if not __m.championNameFromId:
        return False
    return True
