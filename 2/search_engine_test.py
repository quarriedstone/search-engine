import glob
import importlib
import os
import sys
from collections import Counter
from math import isclose

reuters_path = '../reuters21578/'
dir_name = "test_dir"


class HidePrints:  # helper
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


os.chdir(dir_name + "/")
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    ok = '\033[92mOK\033[0m'
    fail = '\033[91mFAIL\033[0m'
    module = importlib.import_module(dir_name + '.' + file[:-3])

    # test build_index
    with HidePrints():
        index = module.build_index(reuters_path, 10)
    if (len(index) == 513 and index['cocoa'] == [7, (1, 7)] and
            index['said'] == [32, (1, 5), (2, 1), (3, 2), (4, 10), (7, 3), (8, 2), (9, 2), (10, 7)]):
        print("build_index", ok)
    else:
        print("build_index", fail)

    # test scoring functions
    query = Counter(module.preprocess("x y z"))
    doc_lengths = {1: 20, 2: 15, 3: 10}
    index = {'x': [2, (1, 1), (2, 1)], 'y': [2, (1, 1), (3, 1)], 'z': [1, (2, 1)]}

    # test cosine_scores
    with HidePrints():
        cosine_scores = module.cosine_scoring(query, doc_lengths, index)
    if cosine_scores[2] > cosine_scores[1] == cosine_scores[3] and isclose(cosine_scores[2], 0.017, abs_tol=1e-3):
        print("cosine_scores", ok)
    else:
        print("cosine_scores", fail)
    # test okapi_scores
    with HidePrints():
        okapi_scores = module.okapi_scoring(query, doc_lengths, index)
    if okapi_scores[2] > okapi_scores[1] > okapi_scores[3] and isclose(okapi_scores[2], 0.653, abs_tol=1e-3):
        print("okapi_scores", ok)
    else:
        print("okapi_scores", fail)
