import glob
import operator
from cmath import isclose

from bs4 import BeautifulSoup
import pickle
import re
import os
from collections import Counter


def extract_categories(path):
    """
    Parses .sgm files in path folder wrt categories each document belongs to.
    Returns a list of documents for each category. One document usually belongs to several categories.
    Categories are contained in special tags (<TOPICS>, <PLACES>, etc.),
    see cat_descriptions_120396.txt file for details
    :param path: original data path
    :return: dict, category:[doc_id1, doc_id2, ...]
    """

    files = glob.glob(path + "*.sgm")
    categories = {}
    for filename in files:

        with open(filename, "rb") as file:
            # Parsing html pages and getting reuters tagged once
            soup = BeautifulSoup(file, "html.parser")
            articles = soup.find_all('reuters')

            for article in articles:

                tags = set(list(map(lambda x: x.get_text(), article.find_all('d'))))
                newid = int(article['newid'])

                for tag in tags:
                    if tag in categories:
                        categories_set = categories[tag]
                        categories_set.add(newid)
                        categories[tag] = categories_set
                    else:
                        categories[tag] = {newid}

    for tag in categories:
        categories[tag] = list(categories[tag])

    return categories


def lm_rank_documents(query, doc_ids, doc_lengths, high_low_index, smoothing, param):
    """
    Scores each document in doc_ids using this document's language model.
    Applies smoothing. Looks up term frequencies in high_low_index
    :param query: dict, term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, doc_id:score
    """

    ld = 0
    for doc in doc_lengths:
        ld += doc_lengths[doc]

    scores = {}
    for doc_id in doc_ids:
        score = 1
        for q in query:
            prob = 0
            if q in high_low_index:
                # Getting high and low indexes
                high, low = high_low_index[q][0], high_low_index[q][1]

                tf = high.get(doc_id, 0) + low.get(doc_id, 0)

                if smoothing is 'additive':
                    prob = (param + tf) / (param * len(high_low_index) + doc_lengths[doc_id])
                else:
                    tf_sum = sum(high.values()) + sum(low.values())
                    prob = param * (tf / doc_lengths[doc_id]) + (1 - param) * (tf_sum / ld)

            score *= prob

        scores[doc_id] = score

    return scores


def lm_define_categories(query, cat2docs, doc_lengths, high_low_index, smoothing, param):
    """
    Same as lm_rank_documents, but here instead of documents we score all categories
    to find out which of them the user is probably interested in. So, instead of building
    a language model for each document, we build a language model for each category -
    (category comprises all documents belonging to it)
    :param query: dict, term:count
    :param cat2docs: dict, category:[doc_id1, doc_id2, ...]
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built last lab
    :param smoothing: which smoothing to apply, either 'additive' or 'jelinek-mercer'
    :param param: alpha for additive / lambda for jelinek-mercer
    :return: dictionary of scores, category:score
    """

    tf_sums = {}
    for t in high_low_index:
        tf_sums[t] = sum(high_low_index[t][0].values()) + sum(high_low_index[t][1].values())
    ld = sum(tf_sums.values())

    scores = {}

    for cat in cat2docs:
        doc_list = cat2docs[cat]
        cat_length = sum([doc_lengths[i] for i in cat2docs[cat]])

        if cat_length is 0:
            scores[cat] = 0
            continue

        score = 1
        for q in query:
            prob = 0
            if q in high_low_index:
                # Getting high and low indexes
                high, low = high_low_index[q][0], high_low_index[q][1]

                tf = 0
                for doc_id in doc_list:
                    tf += high.get(doc_id, 0) + low.get(doc_id, 0)

                if smoothing is 'additive':
                    prob = (param + tf) / (param * len(high_low_index) + cat_length)
                else:
                    pm_d = tf/cat_length

                    pm_c = tf_sums[q]/ld
                    prob = param*pm_d + (1-param)*pm_c

            score *= prob
        scores[cat] = score
    return scores


def extract_categories_descriptions(path):
    """
    Extracts full names for categories, draft version (inaccurate).
    You can use if as a draft for incorporating LM-based scoring to your engine
    :param path: original data path
    :return: dict, category:description
    """
    category2descr = {}
    pattern = r'\((.*?)\)'
    with open(path + 'cat-descriptions_120396.txt', 'r') as f:
        for line in f:
            if re.search(pattern, line) and not (line.startswith('*') or line.startswith('@')):
                category = re.search(pattern, line).group(1)
                if len(category.split()) == 1:
                    category2descr[category.lower()] = line.split('(')[0].strip()
    return category2descr

