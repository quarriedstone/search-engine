import glob
import nltk
from bs4 import BeautifulSoup
import pickle
from collections import Counter, defaultdict
import math
import heapq
import re
import os

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


# tokenize text using nltk lib
def tokenize(text):
    return nltk.word_tokenize(text)


# stem word using provided stemmer
def stem(word, stemmer):
    return stemmer.stem(word)


# checks if word is appropriate - not a stop word and isalpha
def is_apt_word(word):
    return word not in stop_words and word.isalpha()


# combines all previous methods together
def preprocess(text):
    tokenized = tokenize(text.lower())
    return [stem(w, ps) for w in tokenized if is_apt_word(w)]


# Saving dict as .pkl file
def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def build_index(path, limit=None):
    """
    # principal function - builds an index of terms in all documents
    # generates 3 dictionaries and saves on disk as separate files:
    # index - term:[term_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...]
    # doc_lengths - doc_id:doc_length
    # documents - doc_id: doc_content_clean
    :param path: path to directory with original reuters files
    :param limit: number of articles to process, for testing. If limit is not None,
                  return index when done, without writing files to disk
    """

    documents = {}
    doc_lengths = {}
    index = {}
    j = 0  # Counter for articles
    for i in range(0, 22):
        if i >= 10:
            file = open(path + "reut2-0" + str(i) + ".sgm", encoding='latin-1')
        else:
            file = open(path + "reut2-00" + str(i) + ".sgm", encoding='latin-1')

        # Parsing html pages and getting reuters tagged once
        soup = BeautifulSoup(file, "html.parser")
        articles = soup.find_all('reuters')

        for article in articles:

            body = ""
            title = ""
            newid = int(article['newid'])

            try:
                body = article.body.get_text()
            except AttributeError:
                pass

            try:
                title = article.title.get_text()
            except AttributeError:
                pass

            words_list = title + "\n" + body

            # Adding title+body to documents dictionary
            documents[newid] = words_list

            # Processing document and adding document lengths to dictionary
            processed_doc = preprocess(documents[newid])
            doc_lengths[newid] = len(processed_doc)

            # Adding word to index
            for term in processed_doc:
                if term in index:
                    term_freq, docs_dict = index[term]

                    term_freq += 1
                    if newid in docs_dict:
                        docs_dict[newid] += 1
                    else:
                        docs_dict[newid] = 1

                    index[term] = (term_freq, docs_dict)
                else:
                    docs_dict = {newid: 1}
                    index[term] = (1, docs_dict)
            j += 1
            # Checking limit on articles
            if limit is not None:
                if j == limit:
                    break

        # Checking limit on articles
        if limit is not None:
            if j == limit:
                break

    for term in index:
        term_freq, docs_dict = index[term]
        index[term] = [term_freq] + list(docs_dict.items())

    if limit is None:
        save_obj(index, "reuters_index")
        save_obj(documents, "reuters_documents")
        save_obj(doc_lengths, "reuters_doc_lengths")

    return index


def compute_idf_vector(N, index):
    """
    :return: dictionary of idf scores for each term. dict = { newid: {term: idf} }
    """

    dict_vector = {}
    for term in index:
        term_freq, doc_tuples = index[term][0], index[term][1:]
        df_t = len(doc_tuples)

        dict_vector[term] = math.log10(float(N) / df_t)

    return dict_vector


def cosine_scoring(query, doc_lengths, index):
    """
    Computes scores for all documents containing any of query terms
    according to the COSINESCORE(q) algorithm from the book (chapter 6)

    :param query: dictionary - term:frequency
    :return: dictionary of scores - doc_id:score
    """
    idf_dict_vector = compute_idf_vector(len(doc_lengths), index)
    doc_scores = {}

    for q in query:
        if q in idf_dict_vector:
            wt_q = idf_dict_vector[q] * query[q]
        else:
            wt_q = 0

        for tup in index[q][1:]:
            wf_q = idf_dict_vector[q] * tup[1]
            if tup[0] in doc_scores:
                doc_scores[tup[0]] += wt_q * wf_q
            else:
                doc_scores[tup[0]] = wt_q * wf_q

    for doc in doc_scores:
        doc_scores[doc] = doc_scores[doc] / doc_lengths[doc]

    return doc_scores


def okapi_scoring(query, doc_lengths, index, k1=1.2, b=0.75):
    """
    Computes scores for all documents containing any of query terms
    according to the Okapi BM25 ranking function, refer to wikipedia,
    but calculate IDF as described in chapter 6, using 10 as a base of log

    :param query: dictionary - term:frequency
    :return: dictionary of scores - doc_id:score
    """
    # TODO write your code here


def answer_query(raw_query, index, doc_lengths, documents, top_k, scoring_fnc):
    """
    :param raw_query: user query as it is
    :param top_k: how many results to show
    :param scoring_fnc: cosine/okapi
    :return: list of ids of retrieved documents (top_k)
    """
    # pre-process query the same way as documents
    query = preprocess(raw_query)
    # count frequency
    query = Counter(query)
    # retrieve all scores
    scores = scoring_fnc(query, doc_lengths, index)
    # put them in heapq data structure, to allow convenient extraction of top k elements
    h = []
    for doc_id in scores.keys():
        neg_score = -scores[doc_id]
        heapq.heappush(h, (neg_score, doc_id))
    # retrieve best matches
    top_k = min(top_k, len(h))  # handling the case when less than top k results are returned
    print('\033[1m\033[94mANSWERING TO:', raw_query, 'METHOD:', scoring_fnc.__name__, '\033[0m')
    print(top_k, "results retrieved")
    top_k_ids = []
    for k in range(top_k):
        best_so_far = heapq.heappop(h)
        top_k_ids.append(best_so_far)
        article = documents[best_so_far[1]]
        article_terms = tokenize(article)
        intersection = [t for t in article_terms if is_apt_word(t) and stem(t, ps) in query.keys()]
        for term in intersection:  # highlight terms for visual evaluation
            article = re.sub(r'(' + term + ')', r'\033[1m\033[91m\1\033[0m', article, flags=re.I)
        print("-------------------------------------------------------")
        print(article)

    return top_k_ids


def main():
    # reuters_path = 'reuters21578/'
    # if not os.path.isfile('reuters_index.p'):
    #     build_index(reuters_path)
    with open('obj/reuters_index.pkl', 'rb') as fp:
        index = pickle.load(fp)
    with open('obj/reuters_doc_lengths.pkl', 'rb') as fp:
        doc_lengths = pickle.load(fp)
    with open('obj/reuters_documents.pkl', 'rb') as fp:
        documents = pickle.load(fp)
    # answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, cosine_scoring)
    # answer_query("soviet union war afghanistan", index, doc_lengths, documents, 5, okapi_scoring)

    # answer_query("black monday", index, doc_lengths, documents, 5, cosine_scoring)
    # answer_query("black monday", index, doc_lengths, documents, 5, okapi_scoring)

    answer_query("apple personal computer", index, doc_lengths, documents, 5, cosine_scoring)
    # answer_query("apple personal computer", index, doc_lengths, documents, 5, okapi_scoring)


if __name__ == "__main__":
    main()
