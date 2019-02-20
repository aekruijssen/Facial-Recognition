# python search_bing_api.py --query "alan grant" --output dataset/alan_grant
# python search_bing_api.py --query "ian malcolm" --output dataset/ian_malcolm
# python search_bing_api.py --query "ellie sattler" --output dataset/ellie_sattler
# python search_bing_api.py --query "john hammond jurassic park" --output dataset/john_hammond
# python search_bing_api.py --query "owen grady jurassic world" --output dataset/owen_grady
# python search_bing_api.py --query "claire dearing jurassic world" --output dataset/claire_dearing

from requests import exceptions
import argparse
import requests
import cv2
import os

# construct arg parser and parse args
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required=True,
	help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

# set Microsoft Cognitive Services API key w max num results & group size for results
API_KEY = "API_KEY"
MAX_RESULTS = 100
GROUP_SIZE = 50

# set endpoint API URL
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

# store search term in convenience variable, set headers and search pars
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# search
print("[INFO] searching Bing API for '{}'".format(term))
search = requests.get(URL, headers=headers, params=params)
search.raise_for_status()

# grab results from search, incl. total num estimated results from Bing API
results = search.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
	term))

# initialize total num of img downloaded
total = 0

# loop over estimated num results in `GROUP_SIZE` groups
for offset in range(0, estNumResults, GROUP_SIZE):
	# update search pars using current offset, make request for results
	print("[INFO] making request for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
	params["offset"] = offset
	search = requests.get(URL, headers=headers, params=params)
	search.raise_for_status()
	results = search.json()
	print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))

	# loop over results
	for v in results["value"]:
		# download img
		try:
			# request to download img
			print("[INFO] fetching: {}".format(v["contentUrl"]))
			r = requests.get(v["contentUrl"], timeout=30)

			# build path to output img
			ext = v["contentUrl"][v["contentUrl"].rfind("."):]
			p = os.path.sep.join([args["output"], "{}{}".format(
				str(total).zfill(8), ext)])

			# write image
			f = open(p, "wb")
			f.write(r.content)
			f.close()

		except Exception as e:
			if type(e) in EXCEPTIONS:
				print("[INFO] skipping: {}".format(v["contentUrl"]))
				continue

		# load img
		image = cv2.imread(p)

		if image is None:
			print("[INFO] deleting: {}".format(p))
			os.remove(p)
			continue

		# update counter
		total += 1
