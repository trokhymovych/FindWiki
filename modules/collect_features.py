import time
from concurrent.futures import ThreadPoolExecutor
import concurrent

import requests
import mwapi
import pytz
import pandas as pd
import pickle
from datetime import datetime
from copy import deepcopy

session = mwapi.Session("https://en.wikipedia.org", user_agent="ahalfaker@wikimedia.org -- IWSC demo")


def fetch_wp10_score(rev_id):
    response = requests.get('https://ores.wikimedia.org/v3/scores/enwiki/{0}'.format(rev_id))
    try:
        results = response.json()['enwiki']['scores'][str(rev_id)]
        wp10 = results['wp10']['score']
        damaging = results['damaging']['score']["probability"]["true"]
        draftquality = results['draftquality']['score']
        goodfaith = results['goodfaith']['score']["probability"]["true"]
        return wp10, damaging, draftquality, goodfaith
    except:
        return None

# That is a timestamp for 1 July 2017 (FEVER dataset is dated June 2017,
# so we take the latest version of page dated before this time)
max_unix_datetime = 1498856400
tz = pytz.timezone('America/Los_Angeles')
max_time = datetime.fromtimestamp(max_unix_datetime, tz).isoformat()


def get_last_rev_id(name: str, last_date: str = max_time) -> (int, str):
    """
    Function that takes the page name and returns
    :param name, name of the page
    :param last_date, date in ISO 8601 which is used as max date limit
    :return page_id, title, revid, timestamp
    """
    res = session.get(action='query',
                      prop='revisions',
                      titles=name,
                      rvprop=['ids', 'timestamp'],
                      rvlimit=1,
                      rvstart=last_date,
                      formatversion=2, redirects=True)

    page_id = res["query"]["pages"][0]["pageid"]
    title = res["query"]["pages"][0]["title"]
    try:
        revid = res["query"]["pages"][0]['revisions'][0]['revid']
        timestamp = res["query"]["pages"][0]['revisions'][0]['timestamp']
    except:
        revid, timestamp = None, None
    redirected = True if 'redirects' in res["query"] else False

    return name, page_id, title, revid, timestamp, redirected


with open('../data/fever_articles_raw_test.pickle', 'rb') as handle:
    articles = pickle.load(handle)

articles = list(articles)
missing_articles = deepcopy(articles)

not_verified_articles_test = set()
verified_articles_test = dict()


def get_article_features(article):
    try:
        name, page_id, title, revid, timestamp, redirected = get_last_rev_id(article)
        wp10, damaging, draftquality, goodfaith = fetch_wp10_score(revid)

        results = {
            "original_title": name,
            "redirected": redirected,
            "page_id": page_id,
            "title": title,
            "revid": revid,
            "timestamp": timestamp,
            "damaging": damaging,
            "goodfaith": goodfaith,
            "wp10_prediction": wp10["prediction"],
            "wp10_probs": wp10["probability"],
            "draftquality_prediction": draftquality["prediction"],
            "draftquality_probs": draftquality["probability"]
        }
    except:
        results = {
            "original_title": article,
            "redirected": False,
            "page_id": None,
            "title": None,
            "revid": None,
            "timestamp": None,
            "damaging": None,
            "goodfaith": None,
            "wp10_prediction": None,
            "wp10_probs": None,
            "draftquality_prediction": None,
            "draftquality_probs": None
        }

    return results


def get_articles_without_features(pages_features):
    feat = pd.DataFrame(pages_features)
    processed_articles = set(feat[~feat["title"].isna()].original_title.values)
    all_articles = set(feat.original_title.values)
    return list(all_articles - processed_articles)


pages_features = []
start = time.time()
n_iter = 0
while (len(missing_articles) > 0) and (n_iter < 7):
    n_iter += 1
    print(f"Starting iteration {n_iter}. {len(missing_articles)} articles to be processed...")

    start_inner = time.time()
    with ThreadPoolExecutor(10) as executor:
        futures = []
        for article in missing_articles:
            futures.append(executor.submit(get_article_features, article=article))
        for future in concurrent.futures.as_completed(futures):
            pages_features.append(future.result())

    missing_articles = get_articles_without_features(pages_features)
    finish_inner = time.time()

    print("Saving intermediate results ... ")
    features = pd.DataFrame(pages_features)
    features = features[~features["title"].isna()]
    features.to_csv('../data/fever_articles_raw_test_features.csv', index=False)

    print("Done... ")
    print(f"Iteration {n_iter} finished. Time: {finish_inner - start_inner}. "
          f"Missing articles left: {len(missing_articles)}")

finish = time.time()
print("Application finished ", finish - start)
print("Done... ")


