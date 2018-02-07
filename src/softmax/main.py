import champion_info as cinfo
import match_processing as mp

print("")
print("********************************")
print("** Beginning Swain Bot Run! **")
print("********************************")

valid_champ_ids = cinfo.get_champion_ids()
print("Number of valid championIds: {}".format(len(valid_champ_ids)))

reuse_matches = True
if reuse_matches:
    print("Using match data in match_pool.txt.")
    with open('match_pool.txt','r') as infile:
        data = json.load(infile)
    validation_ids = data["validation_ids"]
    training_ids = data["training_ids"]

    n_matches = len(validation_ids) + len(training_ids)
    n_training = len(training_ids)
    training_matches = mp.get_matches_by_id(training_ids)
    validation_matches = mp.get_matches_by_id(validation_ids)

print("***")
print("Validation matches:")
count = 0
for match in validation_matches:
    count += 1
    print("Match: {:2} id: {:4} {:6} vs {:6} winner: {:2}".format(count, match["id"], match["blue_team"], match["red_team"], match["winner"]))
    for team in ["blue", "red"]:
        bans = match[team]["bans"]
        picks = match[team]["picks"]
        pretty_bans = []
        pretty_picks = []
        for ban in bans:
            pretty_bans.append(cinfo.champion_name_from_id(ban[0]))
        for pick in picks:
            pretty_picks.append((cinfo.champion_name_from_id(pick[0]), pick[1]))
        print("{} bans:{}".format(team, pretty_bans))
        print("{} picks:{}".format(team, pretty_picks))
    print("")
print("***")

# Network parameters
state = DraftState(DraftState.BLUE_TEAM,valid_champ_ids)
input_size = state.format_state().shape
output_size = state.num_actions
filter_size = (1024,1024)
regularization_coeff = 7.5e-5#1.5e-4
