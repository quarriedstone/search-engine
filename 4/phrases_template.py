import nltk


def find_ngrams_PMI(tokenized_text, freq_thresh, pmi_thresh, n):
    """
    Finds n-grams in tokenized text, limiting by frequency and pmi value
    :param tokenized_text: list of tokens
    :param freq_thresh: number, only consider ngrams more frequent than this threshold
    :param pmi_thresh: number, only consider ngrams that have pmi value greater than this threshold
    :param n: length of ngrams to consider, can be 2 or 3
    :return: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2'), ... }
    """
    if n == 2:
        bgram_measures = nltk.collocations.BigramAssocMeasures()
        bgram_finder = nltk.BigramCollocationFinder.from_words(tokenized_text)
        bgram_finder.apply_freq_filter(freq_thresh)

        scores = bgram_finder.score_ngrams(bgram_measures.pmi)
        filtered_scores = list(filter(lambda x: x[1] >= pmi_thresh, scores))

        return set([x for x, _ in filtered_scores])

    if n == 3:
        trgram_measures = nltk.collocations.TrigramAssocMeasures()
        trgram_finder = nltk.TrigramCollocationFinder.from_words(tokenized_text)
        trgram_finder.apply_freq_filter(freq_thresh)

        scores = trgram_finder.score_ngrams(trgram_measures.pmi)
        filtered_scores = list(filter(lambda x: x[1] >= pmi_thresh, scores))

        return set([x for x, _ in filtered_scores])

    return set()


def build_ngram_index(tokenized_documents, ngrams):
    """
    Builds index based on ngrams and collection of tokenized docs
    :param tokenized_documents: {doc1_id: ['token1', 'token2', ...], doc2_id: ['token1', 'token2', ...]}
    :param ngrams: set of ngrams tuples - {('ngram1_1', 'ngram1_2'), ('ngram2_1', 'ngram2_2', 'ngram2_3'), ... }
    :return: dictionary - {ngram_tuple :[ngram_tuple_frequency, (doc_id_1, doc_freq_1), (doc_id_2, doc_freq_2), ...], ...}
    """

    return {}
