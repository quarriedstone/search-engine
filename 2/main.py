import glob
import nltk
from bs4 import BeautifulSoup
import pickle
from collections import Counter, defaultdict
import math
import heapq
import re
import os
import time

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()
path = "reuters21578/"
limit = 10


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


start = time.time()

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
    save_obj(doc_lengths, "reuters_doc_length")

end = time.time()
print(end - start)
print(len(index))

print(index["cocoa"])
print(index["said"])
