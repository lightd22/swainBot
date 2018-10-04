import luigi
import requests
import json
import time
import sqlite3
from data.create_database import create_tables
import data.database_ops as dbo
from data.query_wiki import query_wiki

class CreateMatchDB(luigi.Task):
    path_to_db = luigi.Parameter(default="../data/competitiveMatchData.db")

    def output(self):
        return luigi.LocalTarget(self.path_to_db)

    def run(self):
        tableNames = ["game", "pick", "ban", "team"]

        columnInfo = []
        # Game table columns
        columnInfo.append(["id INTEGER PRIMARY KEY",
                            "tournament TEXT","tourn_game_id INTEGER", "week INTEGER", "patch TEXT",
                            "blue_teamid INTEGER NOT NULL", "red_teamid INTEGER NOT NULL",
                            "winning_team INTEGER"])
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

        conn = sqlite3.connect(self.path_to_db)
        cur = conn.cursor()
        print("Creating tables..")
        _ = create_tables(cur, tableNames, columnInfo, clobber = True)
        conn.close()

        return 1

def validate_match_data(match_data):
    """
    validate_match_data performs basic match data validation by examining the following:
        1. Number of picks/bans present in data
        2. Presence of dupicate picks/bans
        3. Duplicate roles on a single side

    Args:
        match_data (dict): dictionary of formatted match data

    Returns:
        bool: True if match_data passes validation checks, False otherwise.
    """
    NUM_BANS = 10
    NUM_PICKS = 10

    is_valid = True
    bans = match_data["bans"]["blue"] + match_data["bans"]["red"]
    picks = match_data["picks"]["blue"] + match_data["picks"]["red"]
    if(len(bans) != NUM_BANS or len(picks)!= NUM_PICKS):
        print("Incorrect number of picks and/or bans found! {} picks, {} bans".format(len(picks), len(bans)))
        is_valid = False

    # Need to consider edge case where teams fail to submit multiple bans (rare, but possible)
    champs = [ban for ban in bans if ban != "none"] + [p for (p,_) in picks]
    if len(set(champs)) != len(champs):
        print("Duplicate submission(s) encountered.")
        counts = {}
        for champ in champs:
            if champ not in counts:
                counts[champ] = 1
            else:
                counts[champ] += 1
        print(sorted([(value, key) for (key, value) in counts.items() if value>1]))
        is_valid = False

    for side in ["blue", "red"]:
        if len(set([pos for (_,pos) in match_data["picks"][side]])) != len(match_data["picks"][side]):
            print("Duplicate position on side {} found.".format(side))
            is_valid = False

    return is_valid

if __name__ == "__main__":
    path_to_db = "../data/competitiveMatchData.db"
    luigi.run(
        cmdline_args=["--path-to-db={}".format(path_to_db)],
        main_task_cls=CreateMatchDB,
        local_scheduler=True)

    conn = sqlite3.connect(path_to_db)
    cur = conn.cursor()

#    deleted_match_ids = [770]
#    dbo.delete_game_from_table(cur, game_ids = deleted_match_ids, table_name="pick")
#    dbo.delete_game_from_table(cur, game_ids = deleted_match_ids, table_name="ban")

    year = "2018"
    schedule = []
    regions = ["EU_LCS","NA_LCS","LPL","LMS","LCK"]; tournaments = ["Spring_Season", "Spring_Playoffs", "Summer_Season", "Summer_Playoffs", "Regional_Finals"]
    schedule.append((regions,tournaments))
    regions = ["NA_ACA","KR_CHAL"]; tournaments = ["Spring_Season", "Spring_Playoffs", "Summer_Season", "Summer_Playoffs"]
    schedule.append((regions,tournaments))
    regions = ["LDL"]; tournaments = ["Spring_Playoffs", "Grand_Finals"]
    schedule.append((regions,tournaments))

    NUM_BANS = 10
    NUM_PICKS = 10
    for regions, tournaments in schedule:
        for region in regions:
            for tournament in tournaments:
                skip_commit = False
                print("Querying: {}".format(year+"/"+region+"/"+tournament))
                gameData = query_wiki(year, region, tournament)
                print("Found {} games.".format(len(gameData)))
                for i,game in enumerate(gameData):
                    is_valid = validate_match_data(game)
                    if not is_valid:
                        skip_commit = True
                        print("Errors in match: h_id {} tourn_g_id {}: {} vs {}".format(game["header_id"], game["tourn_game_id"], game["blue_team"], game["red_team"]))

                if(not skip_commit):
                    print("Attempting to insert {} games..".format(len(gameData)))
                    status = dbo.insert_team(cur,gameData)
                    status = dbo.insert_game(cur,gameData)
                    status = dbo.insert_ban(cur,gameData)
                    status = dbo.insert_pick(cur,gameData)
                    print("Committing changes to db..")
                    conn.commit()
                else:
                    print("Errors found in match data.. skipping commit")
                    raise
