import requests
from . import myRiotApiKey
import time
api_versions = {
    "staticdata": "v3",
    "datadragon": "7.15.1"
}
valid_methods = ["GET", "PUT", "POST"]
region = "na1"
def set_api_key(key):
    """
    Set calling user's API Key

    Args:
        key (string): the Riot API key desired for use.
    """
    myRiotApiKey.api_key = key
    return key

def set_region(reg):
    """
    Set region to run API queries through

    Args:
        reg (string): region through which we are sending API requests
    """
    reg = reg.lower()
    regions = ["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2", "na1", "oce1", "tr1", "ru"]

    assert (reg in regions), "Invalid region!"
    region = reg
    return region

def make_request(request, method, params={}):
    """
    Makes a rate-limited HTTP request to Riot API and returns the response data
    """
    url = "https://{region}.api.riotgames.com/lol/{request}".format(region=region,request=request)
    try:
        response = execute_request(url, method, params)
        if(not response.ok):
            response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Wait and try again on 429 (rate limit exceeded)
        if response.status_code == 429:
            if "X-Rate-Limit-Type" not in e.headers or e.headers["X-Rate-Limit-Type"] == "service":
                # Wait 1 second before retrying
                time.sleep(1)
            else:
                retry_after = 1
                if response.headers["Retry-After"]:
                    retry_after += int(e.headers["Retry-After"])

                time.sleep(retry_after)
            return make_request(request, method, params)
        else:
            raise

def execute_request(url, method, params={}):
    """
    Executes HTTP request using requests library and returns response object.
    Args:
        url (str): full url string to request
        method (str): HTTP method to use (one of "GET", "PUT", or "POST")
        params (dict): dictionary of parameters to send along url

    Returns:
        response object returned by requests
    """

    response = None
    assert(method in valid_methods), "[execute_request] Invalid HTTP method!"
    if(method == "GET"):
        response = requests.get(url=url, params=params)
    return response
