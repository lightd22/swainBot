import random
import json
import sqlite3
from .database_ops import get_matches_by_id, get_game_ids, get_match_data, get_game_ids_by_tournament, get_tournament_data

def test_train_split(n_training, n_validation, path_to_db, list_path=None, save_path=None, match_sources=None, prune_patches=None):
    """
    test_train_split returns a match_ids split into two nominal groups: a training set and a test set.
    Args:
        n_training (int): number of match_ids in training split
        n_test (int): number of match_ids in test_train_split
        path_to_db (str): path to database containing match data
        list_path (str, optional): path to existing match ids to either grow or pruned
        save_path (str, optional): path to save split match ids
        match_sources (dict, optional): dictionary containing "patches" and "tournaments" keys containing lists of tournament and patch identifiers to use for match sources
        prune_patches (list, optional): list of patches to prune from existing split match ids

    Returns:
        Dictionary {"training_ids":list(int),"validation_ids":list(int)}
    """
    save_match_pool = False
    validation_ids = []
    training_ids = []
    if list_path:
        print("Building list off of match data in {}.".format(list_path))
        with open(list_path,'r') as infile:
            data = json.load(infile)
        validation_ids = data["validation_ids"]
        training_ids = data["training_ids"]
        if prune_patches:
            pre_prune_id_count = len(validation_ids)+len(training_ids)
            validation_ids = prune_match_list(validation_ids, path=path_to_db, patches=prune_patches)
            training_ids = prune_match_list(training_ids, path=path_to_db, patches=prune_patches)
            post_prune_id_count = len(validation_ids)+len(training_ids)
            save_match_pool = True
            print("Pruned {} matches from the match list".format(pre_prune_id_count-post_prune_id_count))

    val_diff = max([n_validation - len(validation_ids),0])
    train_diff = max([n_training - len(training_ids),0])

    current = []
    current.extend(validation_ids)
    current.extend(training_ids)

    count = val_diff + train_diff
    if(count > 0):
        new_matches = grow_pool(count, current, path_to_db, match_sources)
        if(val_diff>0):
            print("Insufficient number of validation matches. Attempting to add difference..")
            validation_ids.extend(new_matches[:val_diff])
        if(train_diff>0):
            print("Insufficient number of training matches. Attempting to add difference..")
            training_ids.extend(new_matches[val_diff:count])
        print("Successfully added {} matches to validation and {} matches to training.".format(val_diff, train_diff))
        save_match_pool = True

    if(save_match_pool and save_path):
        print("Saving pool to {}..".format(save_path))
        with open(save_path,'w') as outfile:
            json.dump({"training_ids":training_ids,"validation_ids":validation_ids},outfile)

    return {"training_ids":training_ids,"validation_ids":validation_ids}

def grow_pool(count, current_pool, path_to_db, match_sources=None):
    total = match_pool(0, path_to_db, randomize=False, match_sources=match_sources)["match_ids"]
    new = list(set(total)-set(current_pool))
    assert(len(new) >= count), "Not enough new matches to match required count!"
    random.shuffle(new)

    return new[:count]

def prune_match_list(match_ids, path_to_db, patches=None):
    """
    Prunes match list by removing matches played on specified patches.
    """
    matches = get_matches_by_id(match_ids, path_to_db)
    pruned_match_list = []
    for match in matches:
        patch = match["patch"]
        if patch not in patches:
            pruned_match_list.append(match["id"])
    return pruned_match_list

def match_pool(num_matches, path_to_db, randomize=True, match_sources=None):
    """
    Args:
        num_matches (int): Number of matches to include in the queue (0 indicates to use the maximum number of matches available)
        path_do_db (str): Path to match database to query against
        randomize (bool): Flag for randomizing order of output matches.
        match_sources (dict(string)): Dict containing "tournaments" and "patches" keys to use when building pool, if None, defaults to using patches/tournaments in data/match_sources.json
    Returns:
        match_data (dictionary): dictionary containing two keys:
            "match_ids": list of match_ids for pooled matches
            "matches": list of pooled match data to process

    Builds a set of matchids and match data used during learning phase. If randomize flag is set
    to false this returns the first num_matches in order according to match_sources.
    """
    if(match_sources is None):
        with open("../data/match_sources.json") as infile:
            data = json.load(infile)
            patches = data["match_sources"]["patches"]
            tournaments = data["match_sources"]["tournaments"]
    else:
        patches = match_sources["patches"]
        tournaments = match_sources["tournaments"]

    # If patches or tournaments is empty, grab matches from all patches from specified tournaments or all tournaments from specified matches
    if not patches:
        patches = [None for tournament in tournaments]
    if not tournaments:
        tournaments = [None for patch in patches]

    match_pool = []
    conn = sqlite3.connect(path_to_db)
    cur = conn.cursor()
    # Build list of eligible match ids
    for patch in patches:
        for tournament in tournaments:
            game_ids = get_game_ids(cur, tournament, patch)
            match_pool.extend(game_ids)

    print("Number of available matches for training={}".format(len(match_pool)))
    if(num_matches == 0):
        num_matches = len(match_pool)
    assert num_matches <= len(match_pool), "Not enough matches found to sample!"
    if(randomize):
        selected_match_ids = random.sample(match_pool, num_matches)
    else:
        selected_match_ids = match_pool[:num_matches]

    selected_matches = []
    for match_id in selected_match_ids:
        match = get_match_data(cur, match_id)
        selected_matches.append(match)
    conn.close()
    return {"match_ids":selected_match_ids, "matches":selected_matches}

if __name__ == "__main__":
    match_sources = {"patches":[], "tournaments": ["2018/NA/Summer_Season"]}
    path_to_db = "../data/competitiveMatchData.db"
    out_path = "../data/test_train_split.txt"
    res = test_train_split(20, 20, path_to_db, list_path=out_path, save_path=out_path, match_sources=match_sources, prune_patches=None)
    print(res)
