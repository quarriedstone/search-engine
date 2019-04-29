import pickle

import math
from collections import defaultdict


# Saving dict as .pkl file
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def build_high_low_index(index, freq_thresh):
    """
    Build high-low index based on standard inverted index.
    Based on the frequency threshold, for each term doc_ids are are either put into "high list" -
    if term frequency in it is >= freq_thresh, or in "low list", otherwise.
    high_low_index should return a python dictionary, with terms as keys.
    The structure is different from that of standard index - for each term
    there is a list - [high_dict, low_dict, len(high_dict) + len(low_dict)],
    the latter is document frequency of a term. high_dict, as well as low_dict,
    are python dictionaries, with entries of the form doc_id : term_frequency
    :param index: inverted index
    :param freq_thresh: threshold on term frequency
    :return: dictionary
    """
    index_2 = {}

    for term in index:
        freq, docs = index[term][0], index[term][1:]

        # Making high_dict and low_dict
        high_dict = {}
        low_dict = {}
        for tup in docs:
            if tup[1] >= freq_thresh:
                high_dict[tup[0]] = tup[1]
            else:
                low_dict[tup[0]] = tup[1]

        # Adding high and low dict to index
        index_2[term] = [high_dict, low_dict, len(high_dict) + len(low_dict)]

    save_obj(index_2, "high_low_index")
    return index_2


def filter_docs(query, high_low_index, min_n_docs):
    """
    Return a set of documents in which query terms are found.
    You are interested in getting the best documents for a query, therefore you
    will sequentially check for the following conditions and stop whenever you meet one.
    For each condition also check if number of documents is  >= min_n_docs.
    1) We consider only high lists for the query terms and return a set of documents such that each document contains
    ALL query terms.
    2) We search in both high and low lists, but still require that each returned document should contain ALL query terms.
    3) We consider only high lists for the query terms and return a set of documents such that each document contains
    AT LEAST ONE query term. Actually, a union of high sets.
    4) At this stage we are fine with both high and low lists, return a set of documents such that each of them contains
    AT LEAST ONE query term.

    :param query: dictionary term:count
    :param high_low_index: high-low index you built before
    :param min_n_docs: minimum number of documents we want to receive
    :return: set of doc_ids
    """

    # First case
    app_docs = []
    for term in query:
        if term in high_low_index:
            high_dict = high_low_index[term][0]

            tmp = set()
            for doc_id in high_dict:
                if high_dict[doc_id] >= query[term]:
                    tmp.add(doc_id)
            app_docs.append(tmp)

    app_set = set.intersection(*app_docs)
    if len(app_set) >= min_n_docs:
        return app_set

    # Second case
    app_docs = []
    for term in query:
        if term in high_low_index:
            high_dict, low_dict = high_low_index[term][0], high_low_index[term][1]

            tmp = set()
            for doc_id in high_dict:
                if high_dict[doc_id] >= query[term]:
                    tmp.add(doc_id)

            for doc_id in low_dict:
                if low_dict[doc_id] >= query[term]:
                    tmp.add(doc_id)

            app_docs.append(tmp)

    app_set = set.intersection(*app_docs)
    if len(app_set) >= min_n_docs:
        return app_set

    # Third case
    app_docs = []
    for term in query:
        if term in high_low_index:
            high_dict = high_low_index[term][0]

            tmp = set()
            for doc_id in high_dict:
                tmp.add(doc_id)

            app_docs.append(tmp)

    app_set = set.union(*app_docs)
    if len(app_set) >= min_n_docs:
        return app_set

    # Fourth case
    app_docs = []
    for term in query:
        if term in high_low_index:
            high_dict, low_dict = high_low_index[term][0], high_low_index[term][1]

            tmp = set()
            for doc_id in high_dict:
                tmp.add(doc_id)
            for doc_id in low_dict:
                tmp.add(doc_id)

            app_docs.append(tmp)

    app_set = set.union(*app_docs)
    if len(app_set) >= min_n_docs:
        return app_set

    return set()


def cosine_scoring_docs(query, doc_ids, doc_lengths, high_low_index):
    """
    Change cosine_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    N = len(doc_lengths)
    doc_scores = {}
    for q in query:
        high_list, low_list, df_t = high_low_index[q]

        idf = math.log10(float(N) / df_t)
        if q in high_low_index:
            wt_q = idf * query[q]
        else:
            wt_q = 0

        for doc_id in doc_ids:

            doc_tf = 0
            if doc_id in high_list:
                doc_tf = high_list[doc_id]

            if doc_id in low_list:
                doc_tf = low_list[doc_id]

            wf_q = idf * doc_tf

            if doc_id in doc_scores:
                doc_scores[doc_id] += wt_q * wf_q
            else:
                doc_scores[doc_id] = wt_q * wf_q

    for doc in doc_scores:
        doc_scores[doc] = doc_scores[doc] / doc_lengths[doc]

    return doc_scores


def okapi_scoring_docs(query, doc_ids, doc_lengths, high_low_index, k1=1.2, b=0.75):
    """
    Change okapi_scoring function you built in the second lab
    such that you only score set of doc_ids you get as a parameter,
    and using high_low_index instead of standard inverted index
    :param query: dictionary term:count
    :param doc_ids: set of document ids to score
    :param doc_lengths: dictionary doc_id:length
    :param high_low_index: high-low index you built before
    :return: dictionary of scores, doc_id:score
    """
    scores = defaultdict(int)
    for d in doc_ids:
        scores[d] = 0

    avgdl = sum(doc_lengths.values()) / len(doc_lengths)

    for q in query:
        if q in high_low_index:
            idf = math.log10(len(doc_lengths) / high_low_index[q][2])

            for d, dtf in list(high_low_index[q][0].items()) + list(high_low_index[q][1].items()):
                if d in scores:
                    scores[d] += idf * dtf * (k1 + 1) / (dtf + k1 * (1 - b + b * doc_lengths[d] / avgdl))
    return scores

