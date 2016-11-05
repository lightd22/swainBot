def setApiKey(key):
    """
    Set calling user's API Key

    Args:
        key (string): the Riot API key desired for use.
    """
    return key

def setRegion(region):
    """
    Set region to run API queries through

    Args:
        region (string): region through which we are sending API requests
    """
    region = region.lower()
    regions = ["br", "eune", "euw", "jp", "kr", "lan", "las", "na", "oce", "tr", "ru"]

    assert (region in regions), "Invalid region!"
        
    return region