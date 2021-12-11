import pandas as pd
import grequests
import requests
from tqdm.auto import tqdm
from copy import deepcopy

from utils import ProgressSession

# todo transform into script with args
train = pd.read_csv("../data/fever_train_processed.csv")
test = pd.read_csv("../data/fever_test_processed.csv")
# leave only sentences with evidences
train_filtered = train[train.verifiable == "VERIFIABLE"]
test_filtered = test[test.verifiable == "VERIFIABLE"]


def get_wikipedia_candidates_one(claim):

    print(f"Processing one claim: {claim}")
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": claim,
        "srlimit": 10
    }
    r = requests.get(url=URL, params=params)
    result = r.json()
    results_list = result.get('query', {}).get('search', [])
    return results_list


def get_wikipedia_candidates_async(claims):
    params = [{"action": "query",
               "format": "json",
               "list": "search",
               "srsearch": claim,
               "srlimit": 10}
              for claim in claims]
    URL = "https://en.wikipedia.org/w/api.php"

    with ProgressSession(params) as sess:
        rs = (grequests.get(URL, session=sess, params=param, timeout=1) for param in params)
        results = grequests.map(rs)

    results_list = []
    empty_claims = []

    for r, c in tqdm(zip(results, claims)):
        try:
            result_json = r.json()
            results_list.append(result_json['query']['search'])
        except:
            results_list.append([])
            empty_claims.append(c)

    results_dict = {k: v for k, v in zip(claims, results_list) if len(v) > 0}
    return results_dict, empty_claims


def get_wikipedia_candidates_async_loop(claims, max_n_iter=5, fillna=False):
    empty_claims = deepcopy(claims)
    n_iter = 0
    results_dict_final = dict()
    while (len(empty_claims) > 0) and (n_iter < max_n_iter):
        n_iter += 1
        results_dict, empty_claims = get_wikipedia_candidates_async(empty_claims)
        results_dict_final.update(results_dict)
        print(f"Iteration {n_iter}, number of claims missed: {len(empty_claims)}")

    if fillna:
        results_list = []
        for c in tqdm(claims.values):
            tmp_res = results_dict_final.get(c)
            if tmp_res:
                results_list.append(tmp_res)
            else:
                results_list.append(get_wikipedia_candidates_one(c))
    else:
        results_list = [results_dict_final.get(c, []) for c in tqdm(claims.values)]

    return results_list


candidates_parsed_train = get_wikipedia_candidates_async_loop(train_filtered.claim)
candidates_parsed_test = get_wikipedia_candidates_async_loop(test_filtered.claim)

train_filtered["candidates_parsed"] = candidates_parsed_train
test_filtered["candidates_parsed"] = candidates_parsed_test

train_filtered.to_csv("../data/fever_train_filtered_candidates.csv", index=False)
test_filtered.to_csv("../data/fever_test_filtered_candidates.csv", index=False)

