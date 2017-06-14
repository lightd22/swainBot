import sqlite3
import json
from queryWiki import queryWiki
from championinfo import championIdFromName,championNameFromId
import re
import pandas as pd

regionsDict = {"North_America":"NA", "Europe":"EU", "LCK":"LCK", "LPL":"LPL",
                "LMS":"LMS"}
internationalEventsDict = {"Mid-Season_Invitational":"MSI",
                    "Rift_Rivals":"RR","Season_World_Championship":"WRLDS"}

def tableColInfo(cursor, tableName, printOut=False):
    """
    Returns a list of tuples with column informations:
    (id, name, type, notnull, default_value, primary_key)
    """
    cursor.execute('PRAGMA TABLE_INFO({})'.format(tableName))
    info = cursor.fetchall()

    if printOut:
        print("Column Info:\nID, Name, Type, NotNull, DefaultVal, PrimaryKey")
        for col in info:
            print(col)
    return info

def createTables(cursor, tableNames, columnInfo, clobber = False):
    """
    createTables attempts to create a table for each table in the list tableNames with
    columns as defined by columnInfo. For each if table = tableNames[k] then the columns for
    table are defined by columns = columnInfo[k]. Note that each element in columnInfo must
    be a list of strings of the form column[j] = "jth_column_name jth_column_data_type"

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        tableNames (list(string)): string labels for tableNames
        columnInfo (list(list(string))): list of string labels for each column of each table
        clobber (bool): flag to determine if old tables should be overwritten

    Returns:
        status (int): 0 if table creation failed, 1 if table creation was successful
    """

    for (table, colInfo) in zip(tableNames, columnInfo):
        columnInfoString = ", ".join(colInfo)
        try:
            if clobber:
                cur.execute("DROP TABLE IF EXISTS {tableName}".format(tableName=table))
            cur.execute("CREATE TABLE {tableName} ({columnInfo})".format(tableName=table,columnInfo=columnInfoString))
        except:
            print("Table {} already exists! Here's it's column info:".format(table))
            tableColInfo(cursor, table, True)
            print("***")
            return 0

    return 1

def getTournamentData(gameData):
    """
    getTournamentData cleans up and combines the region/season/split fields in gameData for entry into
    the game table. When combined with the game_id field it uniquely identifies the match played.
    The format of tournamentData output is 'year/split/region_abbrv' (forward slash delimiters)

    Args:
        gameData (dict): dictonary output from queryWiki()
    Returns:
        tournamentData (string): formatted and cleaned region/season/split data
    """
    if gameData["season"] is None:
        year = re.search("([0-9]+)",gameData["region"]).group(0)
    else:
        year = re.search("([0-9]+)",gameData["season"]).group(0)

    if gameData["split"] is None:
        tournamentData = internationalEventsDict["".join(re.split("_?[0-9]+_?",gameData["region"]))]
    else:
        tournamentData = "/".join([gameData["split"],regionsDict[gameData["region"]]])
    tournamentData = "/".join([year,tournamentData])
    return tournamentData

def getGameId(cursor,gameData):
    """
    getGameId looks in the game table for an entry with matching tournament and tourn_game_id as the input
    gameData and returns the id field. If no such entry is found, it adds this game to the game table and returns the
    id field.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (dict): dictionary output from queryWiki()
    Returns:
        gameId (int): Primary key in game table corresponding to this gameData
    """
    tournament = getTournamentData(gameData)
    vals = (tournament,gameData["tourn_game_id"])
    gameId = None
    while gameId is None:
        cursor.execute("SELECT id FROM game WHERE tournament=? AND tourn_game_id=?", vals)
        gameId = cursor.fetchone()
        if gameId is None:
            print("Warning: Game not found. Attempting to add game.")
            err = insertGame(cursor,[game])
        else:
            gameId = gameId[0]
    return gameId

def insertGame(cursor, gameData):
    """
    insertGame attempts to format collected gameData from queryWiki() and insert
    into the game table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): list of dictionary output from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    for game in gameData:
        tournGameId = game["tourn_game_id"] # Which game this is within current split
        tournamentData = getTournamentData(game)

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
                    err = insertTeam(cursor, [game])
                else:
                    blueTeamId = blueTeamId[0]
                    redTeamId = redTeamId[0]

            winner = game["winning_team"]
            vals = (tournamentData, tournGameId, blueTeamId, redTeamId, winner)
            cursor.execute("INSERT INTO game(tournament, tourn_game_id, blue_teamid, red_teamid, winning_team) VALUES(?,?,?,?,?)", vals)
    status = 1
    return status

def insertTeam(cursor, gameData):
    """
    insertTeam attempts to format collected gameData from queryWiki() and insert
    into the team table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        wikiGameData (list(dict)): dictionary output from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    for game in gameData:
        # We don't track all regions (i.e wildcard regions), but they can still appear at
        # international tournaments. When this happens we will track the team, but list their
        # region as NULL.
        if game["split"] is None:
            region = None
        else:
            region = regionsDict[game["region"]]
        teams = [game["blue_team"], game["red_team"]]
        for team in teams:
            vals = (region,team)
            cursor.execute("SELECT * FROM team WHERE display_name=?", (team,))
            result = cursor.fetchone()
            if result is None:
                cursor.execute("INSERT INTO team(region, display_name) VALUES(?,?)", vals)
    status = 1
    return status

def insertBan(cursor, gameData):
    """
    insertBan attempts to format collected gameData from queryWiki() and insert into the
    ban table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): dictionary output from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    teams = ["blue", "red"]
    for game in gameData:
        tournament = getTournamentData(game)
        vals = (tournament,game["tourn_game_id"])
        gameId = getGameId(cursor,game)
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
                    selectionOrder += 1
                    vals = (gameId,championIdFromName(ban),selectionOrder,side)
                    cursor.execute("INSERT INTO ban(game_id, champion_id, selection_order, side_id) VALUES(?,?,?,?)", vals)
    status = 1
    return status

def insertPick(cursor, gameData):
    """
    insertPick formats collected gameData from queryWiki() and inserts it into the pick table of the
    competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        gameData (list(dict)): list of formatted game data from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    assert isinstance(gameData,list), "gameData is not a list"
    teams = ["blue", "red"]
    for game in gameData:
        tournament = getTournamentData(game)
        vals = (tournament,game["tourn_game_id"])
        gameId = getGameId(cursor,game)
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
                    selectionOrder += 1
                    vals = (gameId,championIdFromName(pick),position,selectionOrder,side)
                    cursor.execute("INSERT INTO pick(game_id, champion_id, position_id, selection_order, side_id) VALUES(?,?,?,?,?)", vals)
    status = 1
    return status


if __name__ == "__main__":
    dbName = "competitiveGameData.db"
    tableNames = ["game", "pick", "ban", "team"]

    columnInfo = []
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "tournament TEXT","tourn_game_id INTEGER", "blue_teamid INTEGER NOT NULL",
                        "red_teamid INTEGER NOT NULL", "winning_team INTEGER"])
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "game_id INTEGER", "champion_id INTEGER","position_id INTEGER",
                        "selection_order INTEGER", "side_id INTEGER"])
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "game_id INTEGER", "champion_id INTEGER", "selection_order INTEGER", "side_id INTEGER"])
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "region TEXT", "display_name TEXT"])

    conn = sqlite3.connect("tmp/"+dbName)
    cur = conn.cursor()
    print("Creating tables..")
    createTables(cur, tableNames, columnInfo, clobber = True)


    gameData = queryWiki("League_Championship_Series", "Europe", "2017_Season", "Summer_Season")
    #gameData = queryWiki("2016_Season_World_Championship")

#    print(json.dumps(game, indent=4, sort_keys=True))
#    cid = championIdFromName("khazix")
#    cname = championNameFromId(cid)
#    print("The championId for {} is {}".format(cname,cid))
    print("Attempting to insert {} games..".format(len(gameData)))
    status = insertTeam(cur,gameData)
    status = insertGame(cur,gameData)
    status = insertBan(cur,gameData)
    status = insertPick(cur,gameData)
    print("Committing changes to db..")
    conn.commit()

    query = ("SELECT game.id, game.tourn_game_id, blue.blue_team,red.red_team,game.winning_team"
             "  FROM (game "
             "       JOIN (SELECT id, display_name as blue_team FROM team) AS blue"
             "         ON game.blue_teamid = blue.id)"
             "       JOIN (SELECT id, display_name as red_team FROM team) AS red"
             "         ON game.red_teamid = red.id")
    df = pd.read_sql_query(query, conn)
    print(df)

    query = (
    "SELECT game_id, champion_id, selection_order, side_id"
    "  FROM ban"
    " WHERE game_id = 1 AND side_id = 0"
    )
    df = pd.read_sql_query(query, conn)
    print(df)

    query =(
    "SELECT *  FROM pick LIMIT 20"
    )
    db = pd.read_sql_query(query, conn)
    print(db)

    print("Closing db..")
    conn.close()
