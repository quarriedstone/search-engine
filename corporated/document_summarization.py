from numpy import zeros
from scipy.linalg import svd
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = stopwords.words('english')


def remove_stopwords(sentence):
    """
    Removes stopwords from sentence
    :param sentence: sentence to proceed
    :return: sentence without stop_words
    """
    sentence_new = " ".join([i for i in sentence if i not in stop_words])

    return sentence_new


def calc_rouge1(gold_sentences, rank_sentences):
    """
    Calculates ROUGE 1 score
    :param gold_sentences: sentences given in Opinosis dataset
    :param rank_sentences: predicted best sentences
    :return:
    """
    gold_words = set(" ".join(gold_sentences).replace(".", "").split(" "))
    rank_words = set(" ".join(rank_sentences).replace(".", "").split(" "))

    return len(rank_words.intersection(gold_words)) / len(rank_words)


def read_data(path):
    """
    Reads articles and their summary
    :param path: path to folder with topics and summaries-gold folders
    :return: articles {article_num: article}
    summaries {article_num: {sentence_num: sentence}}
    """
    articles = {}
    summaries = {}
    for i, filename in enumerate(os.listdir(path + "topics")):
        with open(os.path.join("./topics", filename), "r") as f:
            articles[i] = f.read()

    for i, folder_name in enumerate(os.listdir("summaries-gold")):
        gold_file = {}
        for j, filename in enumerate(os.listdir("./summaries-gold/" + folder_name)):
            with open(os.path.join("./summaries-gold/" + folder_name + "/", filename), "r") as f:
                gold_file[j] = f.read()
        summaries[i] = gold_file
    return articles, summaries


def preprocess(text):
    """
    Split and cleans text into sentences
    :param text: text to be preprocessed
    :return: list of preprocessed sentences
    """
    # Getting sentences from text
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]

    # Remove numbers, punctuations, special charecters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # Making sentences lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    # Remove stopwords
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    return clean_sentences


def LSA(sentences):
    """
    Extract 2 most relevant sentences using LSA method
    :param sentences: sentences to proceed
    :return: list with two most relevant sentences
    """

    if len(sentences) == 1:
        return sentences, [0]
    elif len(sentences) == 2:
        return sentences, [0, 1]


    # Building dictionary of words
    word_dict = {}
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for w in words:
            if w in word_dict:
                word_dict[w].append(i)
            else:
                word_dict[w] = [i]

    # Building Count matrix
    keys = [k for k in word_dict.keys() if len(word_dict[k]) > 1]
    keys.sort()
    A = zeros([len(keys), len(sentences)])
    for j, k in enumerate(keys):
        for d in word_dict[k]:
            A[j, d] += 1

    # Making TF-IDF matrix
    words_doc = np.sum(A, axis=0)
    docs_word = np.sum(np.asarray(A > 0, "i"), axis=1)
    rows, cols = A.shape
    for i in range(rows):
        for j in range(cols):
            A[i, j] = (A[i, j] / words_doc[j]) * np.log(float(cols) / docs_word[i])
    A = np.nan_to_num(A)

    # SVD decomposition
    u, s, vh = svd(A)

    # Making first two value absolute
    abs_vt1 = np.absolute(vh[0])
    abs_vt2 = np.absolute(vh[1])

    # Finding first two biggest values
    index1 = np.argmax(abs_vt1)
    summaries = [sentences[index1]]

    index2 = np.argmax(abs_vt2)
    if index2 == index1:
        abs_vt2[index1] = 0
        index2 = np.argmax(abs_vt2)

    summaries.append(sentences[index2])

    return summaries, [index1, index2]


def extract_word_embeddings():
    """
    Extract word embeddings from glove dataset
    :return:
    """
    word_embeddings = {}
    with open('glove.6B.100d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefficients

    return word_embeddings


def TextRank(sentences, word_embeddings):
    """
    Performs TextRank algorithm. Return 2 most informative sentences
    :param sentences: sentences to proceed
    :param word_embeddings:
    :return: list with two most relevant sentences
    """
    # Creating vector of values for given sentences
    sentence_vectors = []
    for i in sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # Building similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    # Calculating similarity matrix
    k = int(len(sentences) * 0.1)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = \
                    cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    # Calculating PageRank
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    # Ranking sentences for summary
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    rank_sentences = []
    for i in range(2):
        rank_sentences.append(ranked_sentences[i][1])

    return rank_sentences


def save_scores_summaries(lsa_rouge1, text_rank_rouge1, text_rank_average, lsa_average):
    """
    Saves summary data in file with format:
        Index of document,
        ROUGE-1 score
        Summary
        Human written summary
    :param lsa_rouge1: dictionary with LSA rouge 1 values
    :param text_rank_rouge1: dictionary with text rank rouge 1 values
    """
    with open("lsa_rouge1.txt", "w") as f:
        for i in lsa_rouge1:
            f.write(str(i) + "\n")
            f.write(str(lsa_rouge1[i][0]) + "\n")
            f.write(str(lsa_rouge1[i][1]) + "\n")
            f.write(str(lsa_rouge1[i][2]) + "\n")
        f.write("\n")
        f.write("LSA average ROUGE-1: " + str(round(np.average(np.array(lsa_average)), 3)) + "\n")
    with open("text_rank_rouge1.txt", "w") as f:
        for i in text_rank_rouge1:
            f.write(str(i) + "\n")
            f.write(str(text_rank_rouge1[i][0]) + "\n")
            f.write(str(text_rank_rouge1[i][1]) + "\n")
            f.write(str(text_rank_rouge1[i][2]) + "\n")
        f.write("\n")
        f.write("TextRank average ROUGE-1: " + str(round(np.average(np.array(text_rank_average)), 3)) + "\n")
    return


def LSA_summary(text):
    cleaned_sentences = preprocess(text)
    summaries, idx = LSA(cleaned_sentences)

    # Getting sentences from text
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]

    if len(idx) == 1:
        return sentences[idx[0]]

    return ".".join([sentences[idx[0]], sentences[idx[1]]])


def main():
    articles, gold_standart = read_data("")

    LSA_average_rouge = []
    TextRank_average_rouge = []

    # Calculating LSA ROUGE-1 scores
    LSA_rouge1 = {}
    print("LSA score calculations")
    for i in range(len(articles)):
        # Getting cleaned sentences for both
        cleaned_sentences = preprocess(articles[i])
        summaries = LSA(cleaned_sentences)

        max_rouge1 = (0, "", "")
        for j in gold_standart[i]:
            gold_sentences = gold_standart[i][j]
            cleaned_gold = preprocess(gold_sentences)
            rouge1 = calc_rouge1(cleaned_gold, summaries)

            # Taking summaries with maximum value
            if rouge1 >= max_rouge1[0]:
                max_rouge1 = (rouge1, summaries, cleaned_gold)

        LSA_rouge1[i] = max_rouge1
        LSA_average_rouge.append(max_rouge1[0])
        print("Document number " + str(i + 1) + ": " + str(max_rouge1[0]))
    print("\n")

    # Calculating TextRank ROUGE-1 scores
    TextRank_rouge1 = {}
    print("TextRank score calculations")
    word_embeddings = extract_word_embeddings()

    for i in range(len(articles)):
        # Getting cleaned sentences for both
        cleaned_sentences = preprocess(articles[i])
        summaries = TextRank(cleaned_sentences, word_embeddings)

        max_rouge1 = (0, "", "")
        for j in gold_standart[i]:
            gold_sentences = gold_standart[i][j]
            cleaned_gold = preprocess(gold_sentences)
            rouge1 = calc_rouge1(cleaned_gold, summaries)

            # Taking summaries with maximum value
            if rouge1 > max_rouge1[0]:
                max_rouge1 = (rouge1, summaries, cleaned_gold)

        TextRank_rouge1[i] = max_rouge1
        TextRank_average_rouge.append(max_rouge1[0])
        print("Document number " + str(i + 1) + ": " + str(max_rouge1[0]))

    # Saving data to file
    save_scores_summaries(LSA_rouge1, TextRank_rouge1, TextRank_average_rouge, LSA_average_rouge)
    print("TextRank average ROUGE-1: " + str(round(np.average(np.array(TextRank_average_rouge)), 3)))
    print("LSA average ROUGE-1: " + str(round(np.average(np.array(LSA_average_rouge)), 3)))
