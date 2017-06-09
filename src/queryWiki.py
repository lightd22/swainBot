import json # JSON tools
import requests # URL api tools
import re # regex tools

def queryWiki(head,*args):
    """
    queryWiki takes identifying sections and subsections for a page title on lol.esportswiki and formats and executes a set of requests to the
    lol.esportswiki's url API looking for the pick/ban data corresponding to the specified sections and subsections. This response is then
    pruned and formatted into a list of dictionaries. Each dictionary corresponds to the pick/ban phase of an LCS game with the following keys:
        "region":
        "season":
        "split":
        "bans": {"blue":, "red"}
        "blue_team":
        "red_team":
        "game_number":
        "picks": {"blue":, "red":}

    Args:
        head (string): first major identifying section for lol.esportswikis page
        *args (string): remaining identifying subsections (in order)
    Returns:
        List of dictionaries containing formatted response data from lol.esportswikis api
    """
    # Common root for all requests
    urlRoot = "http://lol.esportswikis.com/w/api.php"

    # Semi-standardized page suffixes for pick/ban pages
    pageSuffixes = ["", "/4-6", "/7-10", "/2-4", "Bracket_Stage"]

    # Grab region/season/split identifiers
    # If no args given then use head to identify and leave season/split empty
    (region, season, split) = (head, None, None)
    if args:
        (region, season, split) = args

    # Build list of titles of pages to query
    titleRoot = [head]
    for arg in args:
        titleRoot.append(arg)
    titleRoot.append("Picks_and_Bans")
    titleRoot = "/".join(titleRoot)

    titleList = []
    for suffix in pageSuffixes:
        titleList.append(titleRoot+suffix)
    formattedTitleList = "|".join(titleList) # Generate parameter string to pass to API

    params = {"action": "query", "titles": formattedTitleList,
              "prop":"revisions", "rvprop":"content", "format": "json"}

    response = requests.get(url=urlRoot, params=params)
    data = json.loads(response.text)
    pageData = data['query']['pages']
    # Get list of page keys (actually a list of pageIds.. could be used to identify pages?)
    pageKeys = list(pageData.keys())
    pageKeys = [k for k in pageKeys if int(k)>=0] # Filter out "invalid page" and "missing page" responses

    formattedData = []
    gameId = 0
    for page in pageKeys:
        # Get the raw text of the most recent revision of the current page
        rawText = pageData[page]["revisions"][0]["*"]

        # string representation of blue and red teams, ordered by game
        blueTeams = parseRawText("(team1=\w+\s?\w+)",rawText)
        redTeams = parseRawText("(team2=\w+\s?\w+)",rawText)

        # winningTeams holds which team won for each parsed game
        # winner = 1 -> first team won (i.e blue team)
        # winner = 2 -> second team won (i.e red team)
        winningTeams = parseRawText("(winner=[0-9])",rawText)
        winningTeams = [int(i)-1 for i in winningTeams] # Convert string response to int

        # gameNumber holds the number of games played in each match, including the parsed game
        # matches are a collection of games played between the same teams in a "best of" format
        blueScore = parseRawText("(team1score=[0-9])", rawText)
        redScore = parseRawText("(team2score=[0-9])", rawText)
        gameNumber = [int(i)+int(j) for (i,j) in zip(blueScore,redScore)]

        # bans holds the string identifiers of submitted bans for each team in the parsed game
        # ex: bans[i]["blue"] = list of blue team bans for ith game on this page
        blueBans = parseRawText("(blueban[0-9]=\w+\s?\w+)", rawText)
        redBans = parseRawText("(redban[0-9]=\w+\s?\w+)", rawText)

        # picks holds the identifiers of submitted (pick, position) pairs for each team in the parsed game
        # string representation for the positions are converted to ints to match DraftState expectations:
        bluePicks = parseRawText("(bluepick[0-9]=\w+\s?\w+)", rawText)
        bluePickPos = parseRawText("(bluepick[0-9]role=\w+\s?\w+)", rawText)
        bluePicks = list(zip(bluePicks, positionStringToId(bluePickPos)))
        redPicks = parseRawText("(redpick[0-9]=\w+\s?\w+)", rawText)
        redPickPos = parseRawText("(redpick[0-9]role=\w+\s?\w+)", rawText)
        redPicks = list(zip(redPicks, positionStringToId(redPickPos)))

        numGamesOnPage = len(winningTeams)
        if numGamesOnPage > 0: # At least one game found on current page
            picksPerGame = len(bluePicks)//numGamesOnPage
            bansPerGame = len(blueBans)//numGamesOnPage

            for k in range(numGamesOnPage):
                gameId += 1
                bans = {"blue": blueBans[bansPerGame*k:bansPerGame*(k+1)], "red":redBans[bansPerGame*k:bansPerGame*(k+1)]}
                picks = {"blue": bluePicks[picksPerGame*k:picksPerGame*(k+1)], "red":redPicks[picksPerGame*k:picksPerGame*(k+1)]}
                gameData = {"region": region, "season":season, "split": split,
                            "blue_team": blueTeams[k], "red_team": redTeams[k],
                            "winning_team": winningTeams[k], "game_number": gameNumber[k],
                            "bans": bans, "picks": picks, "game_id": gameId}
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
        out.append(re.sub("[^A-Za-z0-9]+", "", rightHandString))  # Remove special chars
    return out

if __name__ == "__main__":
    gameData = queryWiki("League_Championship_Series", "Europe", "2017_Season", "Summer_Season")
    #gameData = queryWiki("2016_Season_World_Championship")
    print("**********************************************")
    print("**********************************************")
    print("Testing queryWiki:")
    print("Number of games found: {}".format(len(gameData)))
    print("**********************************************")
    print("**********************************************")
#    for game in gameData:
#        print(json.dumps(game, indent=4, sort_keys=True))
