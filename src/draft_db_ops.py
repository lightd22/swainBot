import sqlite3
import re
from champion_info import champion_id_from_name,champion_name_from_id, convert_champion_alias, AliasException

regionsDict = {"NA_LCS":"NA", "EU_LCS":"EU", "LCK":"LCK", "LPL":"LPL",
                "LMS":"LMS", "International":"INTL", "NA_ACA": "NA_ACA", "KR_CHAL":"KR_CHAL"}
internationalEventsDict = {"Mid-Season_Invitational":"MSI",
                    "Rift_Rivals":"RR","World_Championship":"WRLDS"}
def get_game_ids_by_tournament(cursor, tournament, patch=None):
    """
    getMatchIdsByTournament queries the connected db for game ids which match the
    input tournament string.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        tournament (string): id string for tournament (ie "2017/EU/Summer_Split")
        patch (string, optional): id string for patch to additionally filter
    Returns:
        gameIds (list(int)): list of gameIds
    """
    if patch:
        query = "SELECT id FROM game WHERE tournament=? AND patch=? ORDER BY id"
        params = (tournament, patch)
    else:
        query = "SELECT id FROM game WHERE tournament=? ORDER BY id"
        params = (tournament,)
    cursor.execute(query, params)
    response = cursor.fetchall()
    vals = []
    for r in response:
        vals.append(r[0])
    return vals

def get_match_data(cursor, gameId):
    """
    get_match_data queries the connected db for draft data and organizes it into a more convenient
    format.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameId (int): primary key of game to process
    Returns:
        match (dict): formatted pick/ban phase data for game
    """
    match = {"id": gameId ,"winner": None, "blue":{}, "red":{}, "blue_team":None, "red_team":None, "week":None, "patch":None}
    # Get winning team
    query = "SELECT tournament, tourn_game_id, week, patch, winning_team FROM game WHERE id=?"
    params = (gameId,)
    cursor.execute(query, params)
    match["tournament"], match["tourn_game_id"], match["week"], match["patch"], match["winner"] = cursor.fetchone()#[0]

    # Get ban data
    query = "SELECT champion_id, selection_order FROM ban WHERE game_id=? and side_id=? ORDER BY selection_order"
    params = (gameId,0)
    cursor.execute(query, params)
    match["blue"]["bans"] = list(cursor.fetchall())

    query = "SELECT champion_id, selection_order FROM ban WHERE game_id=? and side_id=? ORDER BY selection_order"
    params = (gameId,1)
    cursor.execute(query, params)
    match["red"]["bans"] = list(cursor.fetchall())

    # Get pick data
    query = "SELECT champion_id, position_id, selection_order FROM pick WHERE game_id=? AND side_id=? ORDER BY selection_order"
    params = (gameId,0)
    cursor.execute(query, params)
    match["blue"]["picks"] = list(cursor.fetchall())

    query = "SELECT champion_id, position_id, selection_order FROM pick WHERE game_id=? AND side_id=? ORDER BY selection_order"
    params = (gameId,1)
    cursor.execute(query, params)
    match["red"]["picks"] = list(cursor.fetchall())

    query = "SELECT display_name FROM team JOIN game ON team.id = blue_teamid WHERE game.id = ?"
    params = (gameId,)
    cursor.execute(query, params)
    match["blue_team"] = cursor.fetchone()[0]

    query = "SELECT display_name FROM team JOIN game ON team.id = red_teamid WHERE game.id = ?"
    params = (gameId,)
    cursor.execute(query, params)
    match["red_team"] = cursor.fetchone()[0]

    return match

def get_tournament_data(gameData):
    """
    get_tournament_data cleans up and combines the region/year/tournament fields in gameData for entry into
    the game table. When combined with the game_id field it uniquely identifies the match played.
    The format of tournamentData output is 'year/region_abbrv/tournament' (forward slash delimiters)

    Args:
        gameData (dict): dictonary output from query_wiki()
    Returns:
        tournamentData (string): formatted and cleaned region/year/split data
    """
    tournamentData = "/".join([gameData["year"], regionsDict[gameData["region"]], gameData["tournament"]])
    return tournamentData

def get_game_id(cursor,gameData):
    """
    get_game_id looks in the game table for an entry with matching tournament and tourn_game_id as the input
    gameData and returns the id field. If no such entry is found, it adds this game to the game table and returns the
    id field.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (dict): dictionary output from query_wiki()
    Returns:
        gameId (int): Primary key in game table corresponding to this gameData
    """
    tournament = get_tournament_data(gameData)
    vals = (tournament,gameData["tourn_game_id"])
    gameId = None
    while gameId is None:
        cursor.execute("SELECT id FROM game WHERE tournament=? AND tourn_game_id=?", vals)
        gameId = cursor.fetchone()
        if gameId is None:
            print("Warning: Game not found. Attempting to add game.")
            err = insert_game(cursor,[game])
        else:
            gameId = gameId[0]
    return gameId

def delete_game_from_table(cursor, game_ids, table_name):
    """
    Deletes rows corresponding to game_id from table table_name.
    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        game_ids (list(int)): game_ids to be removed from table
        table_name (string): name of table to remove rows from
    Returns:
        status (int): status = 1 if delete was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(game_ids,list), "game_ids is not a list"
    for game_id in game_ids:
        query = "SELECT count(*) FROM {table_name} WHERE game_id=?".format(table_name=table_name)
        vals = (game_id,)
        cursor.execute(query, vals)
        print("Found {count} rows for game_id={game_id} to delete from table {table}".format(count=cursor.fetchone()[0], game_id=game_id, table=table_name))

        query = "DELETE FROM {table_name} WHERE game_id=?".format(table_name=table_name)
        cursor.execute(query, vals)
    status = 1
    return status

def insert_game(cursor, gameData):
    """
    insert_game attempts to format collected gameData from query_wiki() and insert
    into the game table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): list of dictionary output from query_wiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    for game in gameData:
        tournGameId = game["tourn_game_id"] # Which game this is within current tournament
        tournamentData = get_tournament_data(game)

        # Check to see if game data is already in table
        vals = (tournamentData,tournGameId)
        cursor.execute("SELECT id FROM game WHERE tournament=? AND tourn_game_id=?", vals)
        result = cursor.fetchone()
        if result is not None:
            print("game {} already exists in table.. skipping".format(result[0]))
        else:
            # Get blue and red team_ids
            blueTeamId = None
            redTeamId = None
            while (blueTeamId is None or redTeamId is None):
                cursor.execute("SELECT id FROM team WHERE display_name=?",(game["blue_team"],))
                blueTeamId = cursor.fetchone()
                cursor.execute("SELECT id FROM team WHERE display_name=?",(game["red_team"],))
                redTeamId = cursor.fetchone()
                if (blueTeamId is None) or (redTeamId is None):
                    print("*WARNING: When inserting game-- team not found. Attempting to add teams")
                    err = insert_team(cursor, [game])
                else:
                    blueTeamId = blueTeamId[0]
                    redTeamId = redTeamId[0]

            winner = game["winning_team"]
            week = game["week"]
            patch = game["patch"]
            vals = (tournamentData, tournGameId, week, patch, blueTeamId, redTeamId, winner)
            cursor.execute("INSERT INTO game(tournament, tourn_game_id, week, patch, blue_teamid, red_teamid, winning_team) VALUES(?,?,?,?,?,?,?)", vals)
    status = 1
    return status

def insert_team(cursor, gameData):
    """
    insert_team attempts to format collected gameData from query_wiki() and insert
    into the team table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        wikiGameData (list(dict)): dictionary output from query_wiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    for game in gameData:
        # We don't track all regions (i.e wildcard regions), but they can still appear at
        # international tournaments. When this happens we will track the team, but list their
        # region as NULL.
        if game["region"] is "Inernational":
            region = None
        else:
            region = regionsDict[game["region"]]
        teams = [game["blue_team"], game["red_team"]]
        for team in teams:
            vals = (region,team)
            # This only looks for matching display names.. what happens if theres a
            # NA TSM and and EU TSM?
            cursor.execute("SELECT * FROM team WHERE display_name=?", (team,))
            result = cursor.fetchone()
            if result is None:
                cursor.execute("INSERT INTO team(region, display_name) VALUES(?,?)", vals)
    status = 1
    return status

def insert_ban(cursor, gameData):
    """
    insert_ban attempts to format collected gameData from query_wiki() and insert into the
    ban table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): dictionary output from query_wiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    teams = ["blue", "red"]
    for game in gameData:
        tournament = get_tournament_data(game)
        vals = (tournament,game["tourn_game_id"])
        gameId = get_game_id(cursor,game)
        # Check for existing entries in table. Skip if they already exist.
        cursor.execute("SELECT game_id FROM ban WHERE game_id=?",(gameId,))
        result = cursor.fetchone()
        if result is not None:
            print("Bans for game {} already exists in table.. skipping".format(result[0]))
        else:
            for k in range(len(teams)):
                bans = game["bans"][teams[k]]
                selectionOrder = 0
                side = k
                for ban in bans:
                    if ban in ["lossofban","none"]:
                        # Special case if no ban was submitted in game
                        banId = None
                    else:
#                        print("ban={}".format(ban))
                        banId = champion_id_from_name(ban)
                        # If no such champion name is found, try looking for an alias
                        if banId is None:
                            banId = champion_id_from_name(convert_champion_alias(ban))
                    selectionOrder += 1
                    vals = (gameId,banId,selectionOrder,side)
                    cursor.execute("INSERT INTO ban(game_id, champion_id, selection_order, side_id) VALUES(?,?,?,?)", vals)
    status = 1
    return status

def insert_pick(cursor, gameData):
    """
    insert_pick formats collected gameData from query_wiki() and inserts it into the pick table of the
    competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): list of formatted game data from query_wiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    teams = ["blue", "red"]
    for game in gameData:
        tournament = get_tournament_data(game)
        vals = (tournament,game["tourn_game_id"])
        gameId = get_game_id(cursor,game)
        # Check for existing entries in table. Skip if they already exist.
        cursor.execute("SELECT game_id FROM pick WHERE game_id=?",(gameId,))
        result = cursor.fetchone()
        if result is not None:
            print("Picks for game {} already exists in table.. skipping".format(result[0]))
        else:
            for k in range(len(teams)):
                picks = game["picks"][teams[k]]
                selectionOrder = 0
                side = k
                for (pick,position) in picks:
                    if pick in ["lossofpick","none"]:
                        # Special case if no pick was submitted to game (not really sure what that would mean
                        # but being consistent with insert_pick())
                        pickId = None
                    else:
                        pickId = champion_id_from_name(pick)
                        # If no such champion name is found, try looking for an alias
                        if pickId is None:
                            pickId = champion_id_from_name(convert_champion_alias(pick))
                    selectionOrder += 1
                    vals = (gameId,pickId,position,selectionOrder,side)
                    cursor.execute("INSERT INTO pick(game_id, champion_id, position_id, selection_order, side_id) VALUES(?,?,?,?,?)", vals)
    status = 1
    return status
