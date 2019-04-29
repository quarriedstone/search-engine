import numpy as np
from numpy import zeros
import json
import nltk
from nltk.corpus import stopwords
import math
from nltk.corpus import wordnet as wn
from statistics import mean

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
stop_words = stopwords.words('english')
ps = nltk.stem.PorterStemmer()

# Hyper-parameters
# Number of documents to consider relevant in Pseudo-relevance
k = 20
# Number of document to take as train set
t = 600
# Float precision when printing data
precis = 3


def remove_stopwords(sentence):
    """
    Removes stopwords from sentence
    :param sentence: sentence to proceed
    :return: sentence without stop_words
    """
    sentence_new = " ".join([i for i in sentence if i not in stop_words])

    return sentence_new


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


def compute_idf_vector(N, index):
    """
    Computes IDF values for each term
    :return: dictionary of idf scores for each term. dict = { newid: {term: idf} }
    """

    dict_vector = {}
    for term in index:
        term_freq, doc_tuples = index[term][0], index[term][1:]
        df_t = len(doc_tuples)

        dict_vector[term] = np.log(float(N) / df_t)

    return dict_vector


def make_index(documents):
    index = {}
    doc_lengths = {}

    for idn in documents:
        processed_doc = preprocess(documents[idn])
        doc_lengths[idn] = len(processed_doc)

        for term in processed_doc:
            if term in index:
                term_freq, docs_dict = index[term]

                term_freq += 1
                if idn in docs_dict:
                    docs_dict[idn] += 1
                else:
                    docs_dict[idn] = 1

                index[term] = (term_freq, docs_dict)
            else:
                docs_dict = {idn: 1}
                index[term] = (1, docs_dict)

    for term in index:
        term_freq, docs_dict = index[term]
        index[term] = [term_freq] + list(docs_dict.items())

    return index, doc_lengths


def make_query_vector(query, index, doc_lengths):
    """
    Making vector representation of query
    :param query: string
    :param index: inverted index with words
    :param doc_lengths: dictionary of doc_lengths
    :return:
    """
    # Calculating tf in the query
    query = preprocess(query)
    query_tf = {}
    for q in query:
        if q in query_tf:
            query_tf[q] += 1
        else:
            query_tf[q] = 1

    vector_space = []
    N = len(doc_lengths)
    keys = [k for k in index.keys()]
    keys.sort()

    for term in keys:
        # Creating vector for term
        term_vec = zeros(1)

        if term in query:
            term_freq, doc_tuples = index[term][0], index[term][1:]
            idf_t = np.log(float(N) / len(doc_tuples))
            term_vec[0] = query_tf[term] * idf_t

        vector_space.append(term_vec)

    vector_space = np.array(vector_space)

    return vector_space[:, 0]


def make_vector_space(index, doc_lengths):
    """
    Creates TF-IDF vector space from index
    :param index: inverted index
    :param doc_lengths: documents length
    :return: dictionary in form of {term: [vector of TF-IDF values for each document]}
    """

    N = len(doc_lengths)
    vector_space = []
    keys = [k for k in index.keys()]
    keys.sort()

    for term in keys:
        # Creating vector for term
        term_vec = zeros(len(doc_lengths) + 1)

        term_freq, doc_tuples = index[term][0], index[term][1:]
        idf_t = np.log(float(N) / len(doc_tuples))
        for tup in doc_tuples:
            idn = tup[0]
            tf = tup[1]
            term_vec[idn] = tf * idf_t

        vector_space.append(term_vec)

    vector_space = np.array(vector_space)

    return vector_space


def cosine_similarity(query_vector, document_vector):
    """
    Calculates cosine similarity of two vectors
    :param query_vector: query represented as vector
    :param document_vector: document represented as vector
    :return: cosine similarity between vectors
    """
    similarity = np.dot(query_vector, document_vector)
    # Normalization values
    magn1 = np.sqrt(query_vector.dot(query_vector))
    magn2 = np.sqrt(document_vector.dot(document_vector))

    return similarity / (magn1 * magn2)


def relevance_algorithm(vector_query, relevance_list, vector_model, lam=1, beta=0.5, gamma=0.15):
    """
    Rochio algorithm. Uses list of relevant documents to improve query vector
    :param vector_query: query represented as vector
    :param relevance_list: list of relevant documents
    :param vector_model: vector model of document space
    :return: improved vector query
    """
    relevant_docs_id = []
    # Summing up relevant vectors
    relevant_vector_sum = zeros(vector_model.shape[0])
    for doc_id in relevance_list:
        relevant_docs_id.append(doc_id)
        relevant_vector_sum += vector_model[:, doc_id]
    relevant_vector_sum = relevant_vector_sum / len(relevant_docs_id)

    # Summing up non-relevant vectors
    non_relevant_vector_sum = zeros(vector_model.shape[0])
    for i in range(vector_model.shape[1]):
        if i not in relevant_docs_id:
            non_relevant_vector_sum += vector_model[:, i]
    non_relevant_vector_sum = non_relevant_vector_sum / (vector_model.shape[1] - len(relevant_docs_id))

    # Calculating relevance using rochio formula
    improved_vector = lam * vector_query + beta * relevant_vector_sum - gamma * non_relevant_vector_sum

    # We exclude negative values
    improved_vector *= (improved_vector > 0)

    return improved_vector


def pseudo_relevance_algorithm(vector_query, vector_model, k):
    """
    Pseudo relevance algorithm with k number of relevant queries
    :param vector_query: query represented as vector
    :param vector_model: vector model of document space
    :param k: number of documents to consider relevant
    :return: improved vector query
    """
    cosine_train_scores = get_similarity_scores(vector_query, vector_model)
    relevance_list = np.argsort(cosine_train_scores)[-k:]

    improved_vector = relevance_algorithm(vector_query, relevance_list, vector_model)

    return improved_vector


def get_similarity_scores(vector_query, vector_model):
    """
    Gets list of similarities between query and documnets
    :param vector_query: query represented as vector
    :param vector_model: vector model of document space
    :return: list of cosine scores
    """
    cosine_scores = np.zeros(vector_model.shape[1] + 1)
    for idn in range(0, vector_model.shape[1]):
        similarity = cosine_similarity(vector_query, vector_model[:, idn])
        # print("Base: " + str(idn + 1) + ": " + str(similarity))
        if math.isnan(similarity):
            similarity = 0
        cosine_scores[idn] = similarity

    return cosine_scores


def get_relevant_list(range_tup, relevance_tuples):
    """
    Takes relevant list in given range.
    :param range_tup: tuple in form of (start, end)
    :param relevance_tuples: relevant value in tuples in form of (doc_id, relevance)
    :return: list of releavnt document normalized to start value
    """
    relevant_list = []
    for tup in relevance_tuples:
        if range_tup[1] > tup[0] >= range_tup[0]:
            relevant_list.append(int(tup[0]) - range_tup[0])
    return relevant_list


def calc_recall(true_set, calculated_set):
    """
    Calculates recall
    :param true_set: set with all relevant documents
    :param calculated_set: set with predictied relavant documents
    :return: recall score
    """
    TA = true_set.intersection(calculated_set)
    FN = true_set - TA

    if (len(TA) + len(FN)) == 0:
        return 0

    return round(len(TA) / (len(TA) + len(FN)), precis)


def get_top_k_predictions(vector_query, vector_model, k):
    """
    Get top k predictions
    :param vector_query: query represented as vector
    :param vector_model: vector model of document space
    :param k: number of predicted documents
    :return: list relevant documents
    """
    cosine_scores = get_similarity_scores(vector_query, vector_model)
    relevance_list = np.argsort(cosine_scores)[-k:]

    return relevance_list


def get_wordnet_query(query):
    """
    Adds lemmas of hypernyms and hyponyms to query usin wordnet
    :param query: initial query
    :return: updated query
    """

    new_query = set()

    for word in query:
        new_query.add(word)

        for i, synset in enumerate(wn.synsets(word)):
            if i < 2:
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        new_query.add(lemma.name())
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        new_query.add(lemma.name())

    return " ".join(list(new_query))


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
    return round(mean(average_precision), precis)


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
    return round(mean(average_precision), precis)


def main():
    # Reading data
    documents, queries, relevance = read_cranfield("data/")
    # Creating index for data
    index, doc_lengths = make_index(documents)
    # Creating vector space model from index
    vector_model = make_vector_space(index, doc_lengths)
    # Training data
    vector_model_train = vector_model[:, 0:t]
    # Testing data
    vector_model_test = vector_model[:, t:1400]

    # Array with recall values
    non_relevance_recall_values = []
    relevance_recall_values = []
    pseudo_relevance_recall_values = []
    wordnet_recall_values = []

    # Top k results for each query
    top_k_results_non_relevance = []
    top_k_results_relevance = []
    top_k_results_pseudo_relevance = []
    top_k_results_wordnet = []

    # Iterating through each query in Cranfield dataset
    for q_id in range(1, 226):
        # Making TF-IDF vector from query
        vector_query = make_query_vector(queries[q_id], index, doc_lengths)
        wordnet_query = make_query_vector(get_wordnet_query(queries[q_id]), index, doc_lengths)
        # List of relevant documents for training and test sets
        relevance_train = get_relevant_list((0, t), relevance[q_id])
        relevance_test = get_relevant_list((t, 1400), relevance[q_id])

        # Calculating enhanced queries vectors for relevance and pseudo-relevance algorithm
        relevance_query_vector = relevance_algorithm(vector_query, relevance_train, vector_model_train)
        pseudo_relevance_query_vector = pseudo_relevance_algorithm(vector_query, vector_model_train, k=k)

        # Getting top k predictions for non-relevance, relevance and pseudo-relevance feedback
        non_relevance_top_predictions = get_top_k_predictions(vector_query, vector_model_test, len(relevance_test))
        relevance_top_predictions = get_top_k_predictions(relevance_query_vector, vector_model_test,
                                                          len(relevance_test))
        pseudo_relevance_top_predictions = get_top_k_predictions(pseudo_relevance_query_vector, vector_model_test,
                                                                 len(relevance_test))
        wordnet_relevance_top_predictions = get_top_k_predictions(wordnet_query, vector_model_test, len(relevance_test))

        # Calculating recall for each method
        non_relevance_recall_values.append(calc_recall(set(relevance_test), set(non_relevance_top_predictions)))
        relevance_recall_values.append(calc_recall(set(relevance_test), set(relevance_top_predictions)))
        pseudo_relevance_recall_values.append(calc_recall(set(relevance_test), set(pseudo_relevance_top_predictions)))
        wordnet_recall_values.append(calc_recall(set(relevance_test), set(wordnet_relevance_top_predictions)))

        # Adding top k results of query to global list
        top_k_results_non_relevance.append([x + t for x in non_relevance_top_predictions])
        top_k_results_relevance.append([x + t for x in relevance_top_predictions])
        top_k_results_pseudo_relevance.append([x + t for x in pseudo_relevance_top_predictions])
        top_k_results_wordnet.append([x + t for x in wordnet_relevance_top_predictions])

        print("Query id: " + str(q_id))
        print("Query: " + queries[q_id])
        print("Non-relevance recall: " + str(calc_recall(set(relevance_test), set(non_relevance_top_predictions))))
        print("Relevance recall: " + str(calc_recall(set(relevance_test), set(relevance_top_predictions))))
        print(
            "Pseudo-relevance recall: " + str(calc_recall(set(relevance_test), set(pseudo_relevance_top_predictions))))
        print("Wordnet: " + str(calc_recall(set(relevance_test), set(wordnet_relevance_top_predictions))))
        print("\n")

    print("Average scores:")
    print("Non-relevance recall: " + str(round(np.average(np.array(non_relevance_recall_values)), precis)))
    print("Relevance recall: " + str(round(np.average(np.array(relevance_recall_values)), precis)))
    print(
        "Pseudo-relevance recall: " + str(round(np.average(np.array(pseudo_relevance_recall_values)), precis)))
    print("Wordnet: " + str(round(np.average(np.array(wordnet_recall_values)), precis)))
    print("\n")

    print("MAP:")
    print("Non-relevance MAP: " + str(mean_avg_precision(top_k_results_non_relevance, relevance)))
    print("Relevance MAP: " + str(mean_avg_precision(top_k_results_relevance, relevance)))
    print("Pseudo-relevance MAP: " + str(mean_avg_precision(top_k_results_pseudo_relevance, relevance)))
    print("Wordnet MAP: " + str(mean_avg_precision(top_k_results_wordnet, relevance)))
    print("\n")

    print("NDCG:")
    print("Non-relevance NDCG: " + str(NDCG(top_k_results_non_relevance, relevance, 10)))
    print("Relevance NDCG: " + str(NDCG(top_k_results_relevance, relevance, 10)))
    print("Pseudo-relevance NDCG: " + str(NDCG(top_k_results_pseudo_relevance, relevance, 10)))
    print("Wordnet NDCG: " + str(NDCG(top_k_results_wordnet, relevance, 10)))
