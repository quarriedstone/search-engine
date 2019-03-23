import re
import nltk
import pickle

stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
def tokenize(text):
    return nltk.word_tokenize(text)


def is_apt_word(word):
    return word not in stop_words and word.isalpha()


def build_dictionary(documents):
    """
    Build dictionary of original word forms (without stemming, but tokenized, lowercased, and only apt words considered)
    :param documents: dict of documents (contents)
    :return: {'word1': freq_word1, 'word2': freq_word2, ...}

    """
    dictionary = {}
    for ids in documents:
        tokenized = tokenize(documents[ids].lower())
        for word in tokenized:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1

    return dictionary


def build_k_gram_index(dictionary, k):
    """
    Build index of k-grams for dictionary words. Padd with '$' ($word$) before splitting to k-grams
    :param dictionary: dictionary of original words
    :param k: number of symbols in one gram
    :return: {'gram1': ['word1_with_gram1', 'word2_with_gram1', ...],
              'gram2': ['word1_with_gram2', 'word2_with_gram2', ...], ...}
    """
    gram_dict = {}
    for word in dictionary:
        new_word = "$" + word + "$"
        gram_list = [new_word[i:i + k] for i in range(0, len(new_word))]
        # truncating last not k-words
        if len(new_word) % k != 0:
            gram_list = gram_list[:-(len(new_word) % k)]

        for gram in gram_list:
            if gram in gram_dict:
                gram_dict[gram].append(word)
            else:
                gram_dict[gram] = [word]
    return gram_dict


def generate_wildcard_options(wildcard, k_gram_index):
    """
    For a given wildcard return all words matching it using k-grams
    Refer to book chapter 3.2.2
    Don't forget to pad wildcard with '$', when appropriate
    :param wildcard: query word in a form of a wildcard
    :param k_gram_index:
    :return: list of options (matching words)
    """
    new_wildcard = "$" + wildcard + "$"
    k = len(list(k_gram_index.keys())[0])
    gram_list = [new_wildcard[i:i + k] for i in range(0, len(new_wildcard))]
    # truncating last not k-words
    if len(new_wildcard) % k != 0:
        gram_list = gram_list[:-(len(new_wildcard) % k)]

    gram_list_trunc = list(filter(lambda x: x != "$" and "*" not in x, gram_list))

    option_set = set()
    for word in gram_list_trunc:
        if word in k_gram_index:
            if not option_set:
                option_set.update(k_gram_index[word])
            else:
                option_set.intersection_update(k_gram_index[word])
    return list(option_set)


def produce_soundex_code(word):
    """
    Implement soundex algorithm, version from book chapter 3.4
    :param word: word in lowercase
    :return: soundex 4-character code, like 'k450'
    """
    letter_dict = {
        'a': '0', 'e': '0', 'i': '0', 'o': '0', 'u': '0', 'h': '0', 'w': '0', 'y': '0',
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    # Changing letters to digits
    soundex_list = [word[0]]
    for letter in word[1:]:
        soundex_list.append(letter_dict[letter])

    # Removing pairs of consecutive digits
    for i in range(1, len(soundex_list) - 1):
        if soundex_list[i] == soundex_list[i+1]:
            soundex_list[i] = '-'

    # Deleting all zeros and adding trailing zeros
    soundex_word = "".join(list(filter(lambda x: x != '0' and x != '-', soundex_list))) + "0000"

    return soundex_word[:4]

def build_soundex_index(dictionary):
    """
    Build soundex index for dictionary words.
    :param dictionary: dictionary of original words
    :return: {'code1': ['word1_with_code1', 'word2_with_code1', ...],
              'code2': ['word1_with_code2', 'word2_with_code2', ...], ...}
    """
    soundex_dict = {}
    for word in dictionary:
        code = produce_soundex_code(word)
        if code in soundex_dict:
            soundex_dict[code].append(word)
        else:
            soundex_dict[code] = [word]

    return soundex_dict
