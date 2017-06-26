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
        "red_team":
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
    urlRoot = "http://lol.gamepedia.com/api.php"

    # Semi-standardized page suffixes for pick/ban pages
    numWeeks = 12
    pageSuffixes = [""]
    for i in range(numWeeks):
        pageSuffixes.append("/Week_"+str(i+1))
    pageSuffixes.append("/Knockout_Stage")
    groups = ["A", "B", "C", "D"]
    for group in groups:
        pageSuffixes.append("/Group_Stage/"+group)

    # Build list of titles of pages to query
    if region == "International":
        titleRoot = ["_".join([year,tournament])]
    else:
        titleRoot = ["_".join([year,region])]
        titleRoot.append(tournament)
    titleRoot.append("Scoreboards")
    titleRoot = "/".join(titleRoot)

    titleList = []
    for suffix in pageSuffixes:
        titleList.append(titleRoot+suffix)
    formattedTitleList = "|".join(titleList) # Parameter string to pass to API
    params = {"action": "query", "titles": formattedTitleList,
              "prop":"revisions", "rvprop":"content", "format": "json"}

    response = requests.get(url=urlRoot, params=params)
    print(response.url)
    data = json.loads(response.text)
    pageData = data['query']['pages']
    # Get list of page keys (actually a list of pageIds.. could be used to identify pages?)
    pageKeys = list(sorted(pageData.keys()))
    pageKeys = [k for k in pageKeys if int(k)>=0] # Filter out "invalid page" and "missing page" responses
    formattedData = []
    tournGameId = 0
    for page in pageKeys:
        # Get the raw text of the most recent revision of the current page
        # Note that we remove all space characters from the raw text, including those
        # in team or champion names.
        rawText = pageData[page]["revisions"][0]["*"].replace(" ","")

        # string representation of blue and red teams, ordered by game
        blueTeams = parseRawText("(team1=\w+\s?\w+)",rawText)
        redTeams = parseRawText("(team2=\w+\s?\w+)",rawText)

        # winningTeams holds which team won for each parsed game
        # winner = 1 -> first team won (i.e blue team)
        # winner = 2 -> second team won (i.e red team)
        winningTeams = parseRawText("(winner=[0-9])",rawText)
        winningTeams = [int(i)-1 for i in winningTeams] # Convert string response to int
        numGamesOnPage = len(winningTeams)

        if numGamesOnPage > 0: # At least one game found on current page
            # bans holds the string identifiers of submitted bans for each team in the parsed game
            # ex: bans[i]["blue"] = list of blue team bans for ith game on this page
            blueBans = parseRawText("(team1bans=[\w\s',]+)", rawText)
            blueBans = [item.split(",") for item in blueBans]
            redBans = parseRawText("(team2bans=[\w\s',]+)", rawText)
            redBans = [item.split(",") for item in redBans]

            # bluePicks[i][:] = list of picks for ith game by blue team in selection order
            bluePicks = parseRawText("(team1picks=[\w\s',]+)", rawText)
            bluePicks = [item.split(",") for item in bluePicks]
            # bluePicksInLcsOrder[:] = list of all picks made by all blue teams in games on this page
            bluePicksInLcsOrder = parseRawText("(blue[0-9]champion=[\w\s']+)", rawText)

            redPicks = parseRawText("(team2picks=[\w\s',]+)", rawText)
            redPicks = [item.split(",") for item in redPicks]
            redPicksInLcsOrder = parseRawText("(purple[0-9]champion=[\w\s']+)", rawText)

            picksPerGame = len(bluePicksInLcsOrder)//numGamesOnPage
            bansPerGame = len(blueBans)//numGamesOnPage
            for k in range(numGamesOnPage):
                # picks holds the identifiers of submitted (pick, position) pairs for each team in the parsed game
                # string representation for the positions are converted to ints to match DraftState expectations
                print("{} vs {}".format(blueTeams[k],redTeams[k]))
                picks = cleanChampionNames(bluePicks[k][:])
                picksInLcsOrder = cleanChampionNames(bluePicksInLcsOrder[k*picksPerGame:(k+1)*picksPerGame])
                positions = {pick:convertLcsPositions(pos) for pos,pick in enumerate(picksInLcsOrder)}
                bluePickPos = [(pick, positions[pick]) for pick in picks]

                picks = cleanChampionNames(redPicks[k][:])
                picksInLcsOrder = cleanChampionNames(redPicksInLcsOrder[k*picksPerGame:(k+1)*picksPerGame])
                positions = {pick:convertLcsPositions(pos) for pos,pick in enumerate(picksInLcsOrder)}
                redPickPos = [(pick, positions[pick]) for pick in picks]

                tournGameId += 1
                bans = {"blue": blueBans[k], "red":redBans[k]}
                picks = {"blue": bluePickPos, "red":redPickPos}
                gameData = {"region": region, "year":year, "tournament": tournament,
                            "blue_team": blueTeams[k], "red_team": redTeams[k],
                            "winning_team": winningTeams[k],
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
    #gameData = queryWiki("2017", "EU_LCS", "Summer_Split")
    #gameData = queryWiki("2016", "International", "World_Championship")
    gameData = queryWiki("2017", "International", "Mid-Season_Invitational")
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
        bluePicks = []
        bluebans = []
        for ban in bans:
            bluebans.append(ban)
        for (pick,pos) in picks:
            bluePicks.append(pick)
        s = "|team1picks=" + ", ".join(bluePicks)
        t = "|team1bans=" + ", ".join(bluebans)
        print("blue team = {}".format(team1))

        team2 = game["red_team"]
        picks = game["picks"]["red"]
        bans = game["bans"]["red"]
        redPicks = []
        redbans = []
        for ban in bans:
            redbans.append(ban)
        for (pick,pos) in picks:
            redPicks.append(pick)
        t = t + "|team1bans=" + ", ".join(redbans)
        s = s + "|team2picks=" + ", ".join(redPicks)
        print("red team = {}".format(team2))
        print(s)
        print(t)
        print("***")
#    for game in gameData:
#        print(json.dumps(game, indent=4, sort_keys=True))
