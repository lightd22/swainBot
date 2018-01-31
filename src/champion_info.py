import numpy as np
#from cassiopeia import riotapi
from riotapi import make_request, api_versions
import requests
import re
import json
from myRiotApiKey import api_key

# Box is a vacant class with no initial members. This will be used to hold the champion_id list and champion_id <-> name dictionaries.

class Box:
    pass
__m = Box()
__m.champion_name_from_id = None
__m.champion_id_from_name = None
__m.valid_champion_ids = None
__m.championAliases = {
"blitz": "blitzcrank",
"gp": "gangplank",
"jarvan": "jarvaniv",
"cait": "caitlyn",
"lb": "leblanc",
"cass": "cassiopeia",
"casiopeia": "cassiopeia",
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
"tahm": "tahmkench",
"kayne": "kayn",
"zil": "zilean",
"naut": "nautilus",
"morg": "morgana",
"ez": "ezreal"
}

# This is a flag to make champion_info methods look for data stored locally
# rather than query the API. Useful if API is out.
look_local = True

class Champion():
    def __init__(self,dictionary):
        if(look_local):
            # Local file is a cached query to data dragon
            # Data dragon reverses the meaning of keys and ids from the API.
            self.key = dictionary["id"]
            self.id = int(dictionary["key"])
        else:
            self.key = dictionary["key"]
            self.id = int(dictionary["id"])
        self.name = dictionary["name"]
        self.title = dictionary["title"]

class AliasException(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
        self.message = message

def convert_champion_alias(alias):
    """
    Args:
        alias (string): lowercase and pruned string alias for a champion
    Returns:
        name (string): lowercase and pruned string name for champion

    convert_champion_alias converts a given champion alias (ie "blitz")
    and returns the version of that champions proper name which is suitable for passing to
    champion_id_from_name(). If no such alias can be found, this raises an AliasException.

    Example: name = convert_champion_alias("blitz") will yield name = "blitzcrank"
    """
    null_champion = ["none","lossofban"]
    if (alias in null_champion):
        return "none"
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

def champion_name_from_id(champion_id):
    """
    Args:
        champion_id (int): Integer Id corresponding to the desired champion name.
    Returns:
        name (string): String name of requested champion. If no such champion can be found, returns NULL

    champion_name_from_id takes a requested champion_id number and returns the string name of that champion using a champion_name_from_id dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    """
    if __m.champion_name_from_id is None:
       populate_champion_dictionary()

    if (champion_id in __m.champion_name_from_id):
        return __m.champion_name_from_id[champion_id]
    return None

def champion_id_from_name(champion_name):
    """
    Args:
        champion_name (string): lowercase and pruned string label corresponding to the desired champion id.
    Returns:
        id (int): id of requested champion. If no such champion can be found, returns NULL

    champion_id_from_name takes a requested champion name and returns the id label of that champion using a champion_id_from_name dictionary.
    If the dictonary has not yet been populated, this creates the dictionary using cassiopeia's interface to Riot's API.
    Note that champion_name should be all lowercase and have any non-alphanumeric characters (including whitespace) removed.
    """
    if __m.champion_id_from_name is None:
       populate_champion_dictionary()

    if (champion_name in __m.champion_id_from_name):
        return __m.champion_id_from_name[champion_name]
    return None

def valid_champion_id(champion_id):
    """
    Checks to see if champion_id corresponds to a valid champion id code.
    Returns: True if champion_id is valid. False otherwise.
    Args:
        champion_id (int): Id of champion to be verified.
    """
    if __m.champion_name_from_id is None:
       populate_champion_dictionary()

    return champion_id in __m.valid_champion_ids

def get_champion_ids():
    """
    Returns a sorted list of valid champion IDs.
    Args:
        None
    Returns:
        validIds (list(ints)): sorted list of valid champion IDs.
    """
    if __m.valid_champion_ids is None:
        populate_champion_dictionary()

    return __m.valid_champion_ids[:]

def populate_champion_dictionary():
    """
    Args:
        None
    Returns:
        True if succesful, False otherwise
    Populates the module dictionary whose keys are champion Ids and values are strings of the corresponding champion's name.
    """
    #riotapi.set_region("NA")
    #riotapi.set_api_key(api_key)
    #champions = riotapi.get_champions()
    DISABLED_CHAMPIONS = []
    if(look_local):
        with open('champions.json') as local_data:
            response = json.load(local_data)
    else:
        request = "{static}/{version}/champions".format(static="static-data",version=api_versions["staticdata"])
        params = {"locale":"en_US", "dataById":"true", "api_key":api_key }
        response = make_request(request,"GET",params)
    data = response["data"]
    champions = []
    for value in data.values():
        if(value["name"] in DISABLED_CHAMPIONS):
            continue
        champion = Champion(value)
        champions.append(champion)

    __m.champion_name_from_id = {champion.id: champion.name for champion in champions}
    __m.champion_id_from_name = {re.sub("[^A-Za-z0-9]+", "", champion.name.lower()): champion.id for champion in champions}
    __m.valid_champion_ids = sorted(__m.champion_name_from_id.keys())
    if not __m.champion_name_from_id:
        return False
    return True

def create_Champion_fixture():
    valid_ids = get_champion_ids()
    champions = []
    model = 'predict.Champion'
    for cid in valid_ids:
        champion = {}
        champion["model"] = model
        champion["pk"] = cid
        fields = {}
        fields["id"] = cid
        fields["display_name"] = champion_name_from_id(cid)
        champion["fields"] = fields
        champions.append(champion)
    with open('champions_fixture.json','w') as outfile:
        json.dump(champions,outfile)

if __name__ == "__main__":
    create_Champion_fixture()
