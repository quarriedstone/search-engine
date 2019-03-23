import glob
from bs4 import BeautifulSoup
import sys
from collections import deque
import nltk
import sys
import os
import re

# first few functions are just copied from earlier, to avoid imports
# they are for submission only, remove when done and import from your project instead
stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
ps = nltk.stem.PorterStemmer()


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


def build_spimi_index(path, results_path="", n_lines=100, block_size_limit_MB=0.2):
    """
    Builds spimi index - parses .sgm files in the path and handles articles one by one,
    collecting a list of term-doc_id pairs (tuples).
    Before processing next article check if the size of the list >= block_size_limit_MB,
    if yes, then call invert_and_write_to_disk method, which will create index for this block
    and save it to intermediate file.
    When all files finished, call merge_blocks function, to merge all block indexes together.
    Call the resulting file "spimi_index.dat", and intermediate block files "spimi_block_n.dat"

    :param path: path to directory with original reuters files
    :param results_path: where to save the results (spimi index as well as blocks), if not stated, saves in current dir
    :param n_lines: how many lines per block to read simultaneously when merging
    :param block_size_limit_MB: threshold for in-memory size of the term-doc_id pairs list
    :return:
    """
    term_list = []
    block_num = 0
    for filename in os.listdir(path):
        if re.match(r"reut2-[0-9][0-9][0-9]", filename):
            with open(path + filename, encoding='latin-1') as file:

                # Parsing html pages and getting reuters tagged once
                soup = BeautifulSoup(file, "html.parser")
                articles = soup.find_all('reuters')

                for article in articles:
                    if sys.getsizeof(term_list) >= block_size_limit_MB * 1024 * 1024:
                        invert_and_write_to_disk(term_list, results_path, block_num)
                        term_list.clear()
                        block_num += 1

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

                    # Processing document and adding document lengths to dictionary
                    processed_doc = preprocess(words_list)

                    # Adding word to word_list
                    for term in processed_doc:
                        term_list.append((term, newid))

                invert_and_write_to_disk(term_list, results_path, block_num)

    # Merging blocks
    merge_blocks(results_path, n_lines)


def invert_and_write_to_disk(term2doc, results_path, block_num):
    """
    Takes as an input a list of term-doc_id pairs, creates an inverted index out of them,
    sorts alphabetically by terms to allow merge and saves to a block file.
    Each line represents a term and postings list, e.g. abolish 256 1 278 2 295 2
    I.e. term doc_id_1 term_freq_1 doc_id_2 term_freq_2 ...
    See how the file should look like in the test folder
    :param term2doc: list of term-doc_id pairs
    :param results_path: where to save block files
    :param block_num: block number to use for naming a file - 'spimi_block_n.dat', use block_num for 'n'
    """
    index = {}
    for tup in term2doc:
        if tup[0] in index:
            docs_dict = index[tup[0]]

            if tup[1] in docs_dict:
                docs_dict[tup[1]] += 1
            else:
                docs_dict[tup[1]] = 1

            index[tup[0]] = docs_dict
        else:
            docs_dict = {tup[1]: 1}
            index[tup[0]] = docs_dict

    # Making block lines with tuple (term, ((doc_id term freq), ...) ) 
    block_lines_list = []
    for term in index:
        docs_dict = index[term]
        block_lines_list.append((term, sorted((list(docs_dict.items())), key=lambda x: x[0])))

    block_lines_list.sort(key=lambda x: x[0])

    with open(results_path + "spimi_block_%s.dat" % block_num, 'w') as f:
        for tup in block_lines_list:
            line = tup[0]
            for doc in tup[1]:
                line = line + " " + " ".join([str(doc[0]), str(doc[1])])

            f.write(line + "\n")


def merge_blocks(results_path, n_lines):
    """
    This method should merge the intermediate block files.
    First, we open all block files.
    Remember, we are limited in memory consumption,
    so we can simultaneously load only max n_lines from each block file contained in results_path folder.
    Then we find the "smallest" word, and merge all its postings across the blocks.
    Terms are sorted alphabetically, so, it allows to load only small portions of each file.
    When postings for a term are merged, we write it to resulting index file, and go to the next smallest term.
    Don't forget to sort postings by doc_id.
    As necessary, we refill lines for blocks.
    Call the resulting file 'spimi_index.dat'
    See how the file should look like in the test folder
    :param results_path: where to save the resulting index
    :param n_lines: how many lines per block to read simultaneously when merging
    """
    index = {}
    file_list = []
    bool_dict = {}
    file_output = open(results_path + "spimi_index.dat", 'a')

    # Opening files fro directory
    for filename in os.listdir(results_path):
        if re.match(r"spimi_block_[0-9]+.dat", filename):
            file = open(results_path + filename, "r", encoding='latin-1')
            file_list.append(file)
            bool_dict[file] = False

    while 1:
        for f in file_list:
            for i in range(0, n_lines):

                term = f.readline().strip().split()

                if term:
                    doc_list = []

                    for j in range(1, len(term), 2):
                        doc_list.append((int(term[j]), int(term[j + 1])))
                    if term[0] in index:
                        index[term[0]].extend(doc_list)
                    else:
                        index[term[0]] = doc_list
                else:
                    bool_dict[f] = True
                    break

        # If all files are read, then break the while loop
        if all(bool_dict[val] for val in bool_dict):
            break

    # Making block lines with tuple (term, ((doc_id term freq), ...) ) and sorting them
    block_lines_list = []
    for term in index:
        docs_list = index[term]
        docs_list.sort(key=lambda x: x[0])
        block_lines_list.append((term, docs_list))

    block_lines_list.sort(key=lambda x: x[0])

    # Writing line to file
    for tup in block_lines_list:
        line = tup[0]
        for doc in tup[1]:
            line = line + " " + " ".join([str(doc[0]), str(doc[1])])

        file_output.write(line + "\n")

    for file in file_list:
        file.close()

    file_output.close()


def main():
    reuters_orig_path = './test_dir/data/'
    results_path = './test_dir/results/'
    n_lines = 500
    build_spimi_index(reuters_orig_path, results_path, n_lines)


if __name__ == '__main__':
    main()
