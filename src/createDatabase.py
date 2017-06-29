import sqlite3
import json
from queryWiki import queryWiki
from championinfo import championIdFromName,championNameFromId, convertChampionAlias, AliasException
import re
import pandas as pd
import draftDbOps as dbo

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

if __name__ == "__main__":
    dbName = "competitiveGameData.db"
    tableNames = ["game", "pick", "ban", "team"]

    columnInfo = []
    # Game table columns
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "tournament TEXT","tourn_game_id INTEGER", "blue_teamid INTEGER NOT NULL",
                        "red_teamid INTEGER NOT NULL", "winning_team INTEGER"])
    # Pick table columns
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "game_id INTEGER", "champion_id INTEGER","position_id INTEGER",
                        "selection_order INTEGER", "side_id INTEGER"])
    # Ban table columns
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "game_id INTEGER", "champion_id INTEGER", "selection_order INTEGER", "side_id INTEGER"])
    # Team table columns
    columnInfo.append(["id INTEGER PRIMARY KEY",
                        "region TEXT", "display_name TEXT"])

    conn = sqlite3.connect("tmp/"+dbName)
    cur = conn.cursor()
    print("Creating tables..")
    createTables(cur, tableNames, columnInfo, clobber = False)

    regions = ["LPL","LMS","EU_LCS","NA_LCS","LCK"]
    split = "Summer_Season"
    for region in regions:
        print("Querying: {}".format("2017/"+region+"/"+split))
        gameData = queryWiki("2017", region, split)
        for game in gameData:
            seen_bans = set()
            print("{} v {}".format(game["blue_team"], game["red_team"]))
#            print("blue bans: {}".format(game["bans"]["blue"]))
#            print("red bans: {}".format(game["bans"]["red"]))
            bans = game["bans"]["blue"] + game["bans"]["red"]
            for ban in bans:
                if ban not in seen_bans:
                    seen_bans.add(ban)
                else:
                    print(" Duplicate ban found! {}".format(ban))
                    print(seen_bans)

            seen_picks = set()
            for side in ["blue", "red"]:
                seen_positions = set()
                for pick in game["picks"][side]:
                    (p,pos) = pick
                    if p not in seen_picks:
                        seen_picks.add(p)
                    else:
                        print("  Duplicate pick found! {}".format(p))
                        print(seen_picks)

                    if pos not in seen_positions:
                        seen_positions.add(pos)
                    else:
                        print("   Duplicate pos found! {}".format(pos))
                        print(seen_positions)


        print("Attempting to insert {} games..".format(len(gameData)))
        status = dbo.insertTeam(cur,gameData)
        status = dbo.insertGame(cur,gameData)
        status = dbo.insertBan(cur,gameData)
        status = dbo.insertPick(cur,gameData)
        print("Committing changes to db..")
        conn.commit()

#    print(json.dumps(game, indent=4, sort_keys=True))

    query = (
    "SELECT game_id, champion_id, selection_order, side_id"
    "  FROM ban"
    " WHERE game_id = 1 AND side_id = 0"
    )
    df = pd.read_sql_query(query, conn)
    print(df)

    query = ("SELECT game.tournament, game.tourn_game_id, blue.team, red.team, ban.side_id,"
             "       ban.champion_id AS champ, ban.selection_order AS ord"
             "  FROM (game "
             "       JOIN (SELECT id, display_name as team FROM team) AS blue"
             "         ON game.blue_teamid = blue.id)"
             "       JOIN (SELECT id, display_name as team FROM team) AS red"
             "         ON game.red_teamid = red.id"
             "  LEFT JOIN (SELECT game_id, side_id, champion_id, selection_order FROM ban) AS ban"
             "         ON game.id = ban.game_id"
             "  WHERE game.id IN (SELECT game_id FROM ban WHERE champion_id IS ?)")
    params = (None,)
    db = pd.read_sql_query(query, conn, params=params)
    print(db)

    gameIds = dbo.getGameIdsByTournament(cur, "2017/Summer_Season/EU")
    for i in gameIds:
        match = dbo.getMatchData(cur, i)
        print(match)
    print("Closing db..")
    conn.close()
