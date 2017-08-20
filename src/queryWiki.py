import json # JSON tools
import requests # URL api tools
import re # regex tools
from championinfo import convertChampionAlias, championIdFromName

def queryWiki(year, region, tournament):
    """
    queryWiki takes identifying sections and subsections for a page title on leaguepedia and formats and executes a set of requests to the
    API looking for the pick/ban data corresponding to the specified sections and subsections. This response is then
    pruned and formatted into a list of dictionaries. Specified sections and subsections should combine into a unique identifying string
    for a specific tournament and queryWiki() will return all games for that tournament.

    For example, if we are interested in the regular season of the 2017 European Summer Split we would call:
    queryWiki("2017", "EU_LCS", "Summer_Split")

    If we were interested in 2016 World Championship we would pass:
    queryWiki("2016", "International", "World_Championship")

    Each dictionary corresponds to the pick/ban phase of an LCS game with the following keys:
        "region":
        "season":
        "tournament":
        "bans": {"blue":, "red":}
        "blue_team":
        "blue_team_score"
        "red_team":
        "red_team_score:"
        "tourn_game_id":
        "picks": {"blue":, "red":}

    Args:
        year (string): year of game data of interest
        region (string): region of play for games
        tournament (string): which tournament games were played in
    Returns:
        List of dictionaries containing formatted response data from lol.gamepedia api
    """
    # Common root for all requests
    url_root = "https://lol.gamepedia.com/api.php"

    # Semi-standardized page suffixes for pick/ban pages
    page_suffixes = ["", "/Group_Stage", "/Knockout_Stage"]
    for i in range(10):
        page_suffixes.append("/Week_{}".format(i+1))

    formatted_regions = {"NA_LCS":"NA_LCS",
                        "EU_LCS":"EU_LCS",
                        "LCK":"LCK",
                        "LPL":"LPL",
                        "LMS":"LMS"}

    formatted_international_tournaments = {
                        "WRLDS": "World_Championship",
                        "RR/BLUE": "Rift_Rivals/Blue_Rift",
                        "RR/PURPLE": "Rift_Rivals/Purple_Rift",
                        "RR/RED": "Rift_Rivals/Red_Rift",
                        "RR/YELLOW": "Rift_Rivals/Yellow_Rift",
                        "RR/GREEN": "Rift_Rivals/Green_Rift",
                        "MSI": "Mid-Season_Invitational"
    }
    # Build list of titles of pages to query
    if region == "International":
        title_root = ["_".join([year,formatted_international_tournaments[tournament]])]
    else:
        formatted_region = formatted_regions[region]
        formatted_year = "_".join([year,formatted_region])
        title_root = [formatted_year,tournament]
    title_root.append("Scoreboards")
    title_root = "/".join(title_root)

    title_list = []
    for suffix in page_suffixes:
        title_list.append(title_root+suffix)
    formatted_title_list = "|".join(title_list) # Parameter string to pass to API
    params = {"action": "query", "titles": formatted_title_list,
              "prop":"revisions", "rvprop":"content", "format": "json"}

    response = requests.get(url=url_root, params=params)
    print(response.url)
    data = json.loads(response.text)
    page_data = data['query']['pages']
    # Get list of page keys (actually a list of pageIds.. could be used to identify pages?)
    page_keys = list(sorted(page_data.keys()))
    page_keys = [k for k in page_keys if int(k)>=0] # Filter out "invalid page" and "missing page" responses
    formattedData = []
    tournGameId = 0
    for page in page_keys:
        # Get the raw text of the most recent revision of the current page
        # Note that we remove all space characters from the raw text, including those
        # in team or champion names.
        raw_text = page_data[page]["revisions"][0]["*"].replace(" ","").replace("\\n"," ")
        print(page_data[page]["title"])
        # string representation of blue and red teams, ordered by game
        blue_teams = parseRawText("(team1=\w+\s?\w*)",raw_text)
        red_teams = parseRawText("(team2=\w+\s?\w*)",raw_text)

#        blueScores = parseRawText("(team1score=[0-9])",raw_text)
#        redScores = parseRawText("(team2score=[0-9])",raw_text)

        # winning_teams holds which team won for each parsed game
        # winner = 1 -> first team won (i.e blue team)
        # winner = 2 -> second team won (i.e red team)
        winning_teams = parseRawText("(winner=[0-9])",raw_text)
        winning_teams = [int(i)-1 for i in winning_teams] # Convert string response to int
        num_games_on_page = len(winning_teams)

        # bans holds the string identifiers of submitted bans for each team in the parsed game
        # ex: bans[k] = list of bans for kth game on the page
        blue_bans = parseRawText("(team1bans=\w[\w\s'.,]+)", raw_text)
        blue_bans = [blue_bans[k].split(',') for k in range(len(blue_bans))]
        red_bans = parseRawText("(team2bans=\w[\w\s'.,]+)", raw_text)
        red_bans = [red_bans[k].split(',') for k in range(len(red_bans))]

        # blue_picks[i] = list of picks for kth game on the page
        blue_picks = parseRawText("(team1picks=\w[\w\s'.,]+)", raw_text)
        blue_picks = [blue_picks[k].split(',') for k in range(len(blue_picks))]
        red_picks = parseRawText("(team2picks=\w[\w\s'.,]+)", raw_text)
        red_picks = [red_picks[k].split(',') for k in range(len(red_picks))]

        picks_per_game = len(blue_picks[0])
        bans_per_game = len(blue_bans[0])

        # blue_picks_in_lcs_order initially raw list of picks in lcs order (not sorted by match)
        blue_picks_in_lcs_order = parseRawText("(blue[0-9]champion=[\w\s'.,]+)",raw_text)
        blue_picks_in_lcs_order = [blue_picks_in_lcs_order[picks_per_game*k:picks_per_game*(k+1)] for k in range(num_games_on_page)]
        blue_position_dicts = list(map(createPositionDict,blue_picks_in_lcs_order))
        #print(blue_position_dicts[11].keys())
        red_picks_in_lcs_order = parseRawText("(purple[0-9]champion=[\w\s'.,]+)",raw_text)
        red_picks_in_lcs_order = [red_picks_in_lcs_order[picks_per_game*k:picks_per_game*(k+1)] for k in range(num_games_on_page)]
        red_position_dicts = list(map(createPositionDict,red_picks_in_lcs_order))
        #print(red_position_dicts[11].keys())
        total_blue_bans = sum([len(bans) for bans in blue_bans])
        total_red_bans = sum([len(bans) for bans in red_bans])
        total_blue_picks = sum([len(picks) for picks in blue_picks])
        total_red_picks = sum([len(picks) for picks in red_picks])

        print("Total number of games found: {}".format(num_games_on_page))
        print("There should be {} bans. We found {} blue bans and {} red bans".format(num_games_on_page*5,total_blue_bans,total_red_bans))
        print("There should be {} picks. We found {} blue picks and {} red picks".format(num_games_on_page*5,total_blue_picks,total_red_picks))
        assert total_red_bans==total_blue_bans, "Bans don't match!"
        assert total_red_picks==total_blue_picks, "Picks don't match!"
        if num_games_on_page > 0: # At least one game found on current page
            for k in range(num_games_on_page):
                # submissions holds the identifiers of submitted (pick, position) pairs for each team in the parsed game
                # string representation for the positions are converted to ints to match DraftState expectations
                print("Game {}: {} vs {}".format(k+1,blue_teams[k],red_teams[k]))
                picks = cleanChampionNames(blue_picks[k])
                positions = [blue_position_dicts[k][pick] for pick in picks]
                blue_submissions = [(picks[k], positions[k]) for k in range(picks_per_game)]

                picks = cleanChampionNames(red_picks[k])
                positions = [red_position_dicts[k][pick] for pick in picks]
                red_submissions = [(picks[k], positions[k]) for k in range(picks_per_game)]

                tournGameId += 1
                bans = {"blue": cleanChampionNames(blue_bans[k]), "red":cleanChampionNames(red_bans[k])}
                picks = {"blue": blue_submissions, "red":red_submissions}
                gameData = {"region": region, "year":year, "tournament": tournament,
                            "blue_team": blue_teams[k], "red_team": red_teams[k],
                            "winning_team": winning_teams[k],
                            "blue_score":-1, "red_score":-1,
                            "bans": bans, "picks": picks, "tourn_game_id": tournGameId}
                formattedData.append(gameData)

    return formattedData
def positionStringToId(positions):
    """
    positionStringToId converts input position strings to their integer representations defined by:
        Position 1 = Primary farm (ADC)
        Position 2 = Secondary farm (Mid)
        Position 3 = Tertiary farm (Top)
        Position 4 = Farming support (Jungle)
        Position 5 = Primary support (Support)
    Note that because of variable standardization of the string representations for each position
    (i.e "jg"="jng"="jungle"), this function only looks at the first character of each string when
    assigning integer positions since this seems to be more or less standard.

    Args:
        positions (list(string))
    Returns:
        list(int)
    """

    d = {"a":1, "m":2, "t":3, "j":4, "s":5} # This is lazy and I know it
    out = []
    for position in positions:
        char = position[0] # Look at first character for position information
        out.append(d[char])
    return out

def parseRawText(regex, rawText):
    """
    parseRawText is a helper function which outputs a list of matching expressions defined
    by the regex input. Note that this function assumes that each regex yields matches of the form
    "A=B" which is passed to splitIdStrings() for fomatting.

    Args:
        regex: desired regex to match with
        rawText: raw input string to find matches in
    Returns:
        List of formatted strings containing the matched data.
    """
    # Parse raw text responses for data. Note that a regular expression of the form
    # "(match)" will produce result = [stuff_before_match, match, stuff_after_match]
    # this means that the list of desired matches will be result[1::2]
    out = re.split(regex, rawText)
    out = splitIdStrings(out[1::2]) # Format matching strings
    return out

def splitIdStrings(rawStrings):
    """
    splitIdStrings takes a list of strings each of the form "A=B" and splits them
    along the "=" delimiting character. Returns the list formed by each of the "B"
    components of the raw input strings. For standardization purposes, the "B" string
    has the following done to it:
        1. replace uppercase characters with lowercase
        2. remove special characters (i.e non-alphanumeric)

    Args:
        rawStrings (list of strings): list of strings, each of the form "A=B"
    Returns:
        out (list of strings): list of strings formed by the "B" portion of each of the raw input strings
    """
    out = []
    for string in rawStrings:
        rightHandString = string.split("=")[1].lower() # Grab "B" part of string, make lowercase
        out.append(re.sub("[^A-Za-z0-9,]+", "", rightHandString))  # Remove special chars
    return out

def convertLcsPositions(index):
    """
    Given the index of a pick in LCS order, returns the position id corresponding
    to that index.

    LCS picks are submitted in the following order
    Index | Role | Position
    0       Top    3
    1       Jng    4
    2       Mid    2
    3       Adc    1
    4       Sup    5
    """
    lcsOrderToPos = {i:j for i,j in enumerate([3,4,2,1,5])}
    return lcsOrderToPos[index]

def createPositionDict(picks_in_lcs_order):
    """
    Given a list of champions selected in lcs order (ie top,jungle,mid,adc,support)
    returns a dictionary which matches pick -> position.
    Args:
        picks_in_lcs_order (list(string)): list of string identifiers of picks. Assumed to be in LCS order
    Returns:
        dict (dictionary): dictionary with champion names for keys and position that the key was played in for value.
    """
    d = {}
    cleaned_names = cleanChampionNames(picks_in_lcs_order)
    for k in range(len(picks_in_lcs_order)):
        pos = convertLcsPositions(k)
        d.update({cleaned_names[k]:pos})
    return d


def cleanChampionNames(names):
    """
    Takes a list of champion names and standarizes them by looking for possible aliases
    if necessary.
    Args:
        names (list(string)): list of champion names to be standardized
    Returns:
        cleanedNames (list(string)): list of standardized champion names
    """
    cleanedNames = []
    for name in names:
        if championIdFromName(name) is None:
            name = convertChampionAlias(name)
        cleanedNames.append(name)
    return cleanedNames

if __name__ == "__main__":
    gameData = queryWiki("2017", "EU_LCS", "Summer_Split")
    #gameData = queryWiki("2016", "International", "WRLDS")
    #gameData = queryWiki("2017", "International", "MSI")
    print("**********************************************")
    print("**********************************************")
    print("Testing queryWiki:")
    print("Number of games found: {}".format(len(gameData)))
    print("**********************************************")
    print("**********************************************")
    for game in gameData:
        team1 = game["blue_team"]
        picks = game["picks"]["blue"]
        bans = game["bans"]["blue"]
        print(bans)
        blue_picks = []
        blue_bans = []
        for ban in bans:
            if ban is None:
                blue_bans.append("None")
            else:
                blue_bans.append(ban)
        for (pick,pos) in picks:
            blue_picks.append(pick)
        s = "|team1picks=" + ", ".join(blue_picks)
        t = "|team1bans=" + ", ".join(blue_bans)
        print("blue team = {}".format(team1))

        team2 = game["red_team"]
        picks = game["picks"]["red"]
        bans = game["bans"]["red"]
        red_picks = []
        red_bans = []
        for ban in bans:
            if ban is None:
                red_bans.append("None")
            else:
                red_bans.append(ban)
        for (pick,pos) in picks:
            red_picks.append(pick)
        t = t + "|team2bans=" + ", ".join(red_bans)
        s = s + "|team2picks=" + ", ".join(red_picks)
        print("red team = {}".format(team2))
        print(s)
        print(t)
        print("***")
#    for game in gameData:
#        print(json.dumps(game, indent=4, sort_keys=True))
