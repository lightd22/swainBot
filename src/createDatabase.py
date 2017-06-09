import sqlite3
import json
from queryWiki import queryWiki
from championinfo import championIdFromName,championNameFromId
import re

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

def createTables(cursor, tableNames, columnInfo):
    """
    createTables attempts to create a table for each table in the list tableNames with
    columns as defined by columnInfo. For each if table = tableNames[k] then the columns for
    table are defined by columns = columnInfo[k]. Note that each element in columnInfo must
    be a list of strings of the form column[j] = "jth_column_name jth_column_data_type"

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        tableNames (list(string)): string labels for tableNames
        columnInfo (list(list(string))): list of string labels for each column of each table

    Returns:
        status (int): 0 if table creation failed, 1 if table creation was successful
    """

    for (table, colInfo) in zip(tableNames, columnInfo):
        columnInfoString = ", ".join(colInfo)
        try:
            cur.execute("DROP TABLE IF EXISTS {tableName}".format(tableName=table))
            cur.execute("CREATE TABLE {tableName} ({columnInfo})".format(tableName=table,columnInfo=columnInfoString))
        except:
            print("Table {} already exists! Here's it's column info:".format(table))
            tableColInfo(cursor, table, True)
            print("***")
            return 0

    return 1

def insertGame(cursor, gameData):
    """
    insertGame attempts to format collected gameData from queryWiki() and insert
    into the game table in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        wikiGameData (list(dict)): dictionary output from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    # Do some minor clean up / manipulation to format gameData for row entry.
    # If the "season" and "split" values are empty strings then the rows added are
    # from an international event and require slightly different handling.
    gameId = gameData["game_id"]
    if gameData["season"] is None:
        year = re.search("([0-9]+)",gameData["region"]).group(0)
    else:
        year = re.search("([0-9]+)",gameData["season"]).group(0)

    if gameData["split"] is None:
        split = internationalEventsDict["".join(re.split("_?[0-9]+_?",gameData["region"]))]
    else:
        split = "/".join([gameData["split"],regionsDict[gameData["region"]]])

    split = "/".join([year,split])
    cursor.execute("SELECT id FROM team WHERE display_name=?",(gameData["blue_team"],))
    blueTeamId = cursor.fetchone()
    cursor.execute("SELECT id FROM team WHERE display_name=?",(gameData["red_team"],))
    redTeamId = cursor.fetchone()
    if (blueTeamId is None) or (redTeamId is None):
        print("*ERROR: When inserting-- team not found!")
        return status
    else:
        blueTeamId = blueTeamId[0]
        redTeamId = redTeamId[0]
    winner = gameData["winning_team"]
    vals = (split, gameId, blueTeamId, redTeamId, winner)
    cursor.execute("INSERT INTO game(split, game_number, blue_teamid, red_teamid, winning_team) VALUES(?,?,?,?,?)", vals)
    cursor.execute("SELECT * FROM game")
    print(cursor.fetchone())
    status = 1
    return status

def insertTeam(cursor, gameData):
    """
    insertTeam attempts to format collected gameData from queryWiki() and insert
    into the game team in the competitiveGameData.db.

    Args:
        cursor (sqlite cursor): cursor used to execute commmands
        wikiGameData (list(dict)): dictionary output from queryWiki()
    Returns:
        status (int): status = 1 if insert was successful, otherwise status = 0
    """
    status = 0
    if gameData["split"] is None:
        region = None
    else:
        region = regionsDict[gameData["region"]]
    teams = [gameData["blue_team"], gameData["red_team"]]
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
    """
    status = 0
    bans = gameData["bans"]["blue"]
    status = 1
    return status
if __name__ == "__main__":
    dbName = "competitiveGameData.db"
    tableNames = ["game", "pick", "ban", "team"]

    columnInfo = []
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "split TEXT","game_number INTEGER", "blue_teamid INTEGER NOT NULL",
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
    createTables(cur, tableNames, columnInfo)
    print("Committing changes to db..")

    gameData = queryWiki("League_Championship_Series", "Europe", "2017_Season", "Summer_Season")
    #gameData = queryWiki("2016_Season_World_Championship")
    game = gameData[0]
    print(json.dumps(game, indent=4, sort_keys=True))
    cid = championIdFromName("khazix")
    cname = championNameFromId(cid)
    print("The championId for {} is {}".format(cname,cid))
    status = insertTeam(cur,game)
    status = insertGame(cur,game)
    conn.commit()
    conn.close()
