import math
import numpy as np
import matplotlib.pyplot as plt
import json
from statistics import mean


def interp(pr, rec):
    i = 0
    while i < len(pr) and pr[i][0] < rec:
        i += 1
    if i == len(pr):
        return None

    sin_pr = 0
    for noth, current in pr[i:]:
        sin_pr = max(sin_pr, current)
    return sin_pr


def eleven_points_interpolated_avg(top_k_results, relevance, plot=True):
    """
    Returns 11-points interpolated average over all queries. Refer to chapter 8.4 for explanation.
    First calculate values of precision-recall curve for each query, interpolate them, and average over all queries.
    This function is intended to use when for each query all documents are scored until the last relevant element
    is met. Because we don't usually score each document, instead retrieving only top-k results, we will adapt
    this function. Concretely, if for some query no results are retrieved for some recall level onward
    (e.g. starting with 0.7), then we only count in available values, ignoring the rest. In other words, for each
    recall level we average only over those queries for which precision is available at this recall level.
    Treats relevance judgments as binary - either relevant or not.
    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :param plot: whether to plot the graph or not
    :return: interpolated_avg, list of 11 values
    """
    interpolated_avg = []

    # Making 11 placeholders for avg values
    for i in range(11):
        interpolated_avg.append([])

    for i in range(0, len(top_k_results)):

        relevant_results = set()
        for doc in relevance[i + 1]:
            relevant_results.add(doc[0])

        pr = []

        relevant_docs = 0
        for j in range(len(top_k_results[i])):
            if top_k_results[i][j] in relevant_results:
                relevant_docs += 1
            pr += [(relevant_docs / len(relevant_results), relevant_docs / (j + 1))]
        pr.sort()

        rec = 0
        for j in range(0, 11):
            tmp = interp(pr, rec)

            if tmp is not None:
                interpolated_avg[j].append(tmp)
            rec += 0.1

    for i in range(0, len(interpolated_avg)):
        interpolated_avg[i] = mean(interpolated_avg[i])

    if plot:
        X = np.linspace(0, 1, 11)
        plt.plot(X, interpolated_avg)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()

    return interpolated_avg


def mean_avg_precision(top_k_results, relevance):
    """
    Calculates MAP score for search results, treating relevance judgments as binary - either relevant or not.
    Refer to chapter 8.4 for explanation
    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :return: calculated MAP score
    """
    average_precision = []
    for i in range(0, len(top_k_results)):
        relevant_results = set()
        for doc in relevance[i + 1]:
            relevant_results.add(doc[0])
        prec = []
        relevant_docs = 0
        for j in range(0, len(top_k_results[i])):
            if top_k_results[i][j] in relevant_results:
                relevant_docs += 1
                prec += [relevant_docs / (j + 1)]
        average_precision += [mean(prec) if len(prec) > 0 else 0]
    return mean(average_precision)


def NDCG(top_k_results, relevance, top_k):
    """
    Computes NDCG score for search results (again chapter 8.4). Here relevance is not considered as binary - the bigger
    the judgement score is, the more relevant is the document to a query. Because in our cranfield dataset relevance
    judgements are presented in a different way (1 is most relevant, 4 is least), we will invert it, replacing each
    score with (5-score). For example, if the score was 2, it becomes 5-2=3.
    To find normalization factor for each query, think in this direction - for this particular query what would be an
    ideal DCG score? What documents should have (ideally) been returned by the search engine to maximize the DCG score?
    When you find it, just normalize the real DCG score by ideal DCG score, that's it.
    :param top_k_results: list of lists of ranked results for each query [[doc_id1, doc_id2,...], ...]
                          the i-th result corresponds to (i+1)-th query_id. There may be less than top_k
                          results returned for a query, but never more.
    :param relevance: dict, query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    :param top_k: (max) number of results retrieved for each query, use this value to find normalization
                  factor for each query
    :return: NDCG score
    """
    average_precision = []
    for i in range(0, len(top_k_results)):
        relevant_results = dict(relevance[i + 1])

        dsg = 0
        for j in range(0, len(top_k_results[i][:top_k])):

            tmp_doc = top_k_results[i][:top_k][j]
            if tmp_doc in relevant_results:
                r = 5 - relevant_results[tmp_doc]
                dsg += (math.pow(2, r) - 1) / math.log2(j + 2)

        results = set(top_k_results[i])
        for doc in relevant_results.keys():
            results.add(doc)

        results = list(results)
        results.sort(key=lambda doc: 5 - relevant_results[doc] if doc in relevant_results else 0, reverse=True)

        idsg = 0
        for j in range(0, len(results[:top_k])):

            tmp_doc = results[:top_k][j]
            if tmp_doc in relevant_results:
                r = 5 - relevant_results[tmp_doc]
                idsg += (math.pow(2, r) - 1) / math.log2(j + 2)

        average_precision += [dsg / idsg if idsg != 0 else 0]
    return mean(average_precision)


def read_cranfield(path):
    """
    Helper function, parses Cranfield data. Used for tests. Use it to evaluate your own search engine
    :param path: original data path
    :return: dictionaries - documents, queries, relevance
    relevance comes in form of tuples - query_id:[(relevant_doc_id1, score1), (relevant_doc_id2, score2), ...]
    """
    documents = {}
    queries = {}
    relevance = {}
    for doc in json.load(open(path + 'cranfield_data.json')):
        documents[doc['id']] = doc['body']
    for query in json.load(open(path + 'cran.qry.json')):
        queries[query['query number']] = query['query']
    for rel in json.load(open(path + 'cranqrel.json')):
        query_id = int(rel['query_num'])
        doc_id = int(rel['id'])
        if query_id in relevance:
            relevance[query_id].append((doc_id, rel['position']))
        else:
            relevance[query_id] = [(doc_id, rel['position'])]
    return documents, queries, relevance
