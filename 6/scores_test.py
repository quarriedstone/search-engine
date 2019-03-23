import glob
import importlib
import os
import sys
import pickle
from collections import Counter
from math import isclose


class Checker:
    ok = '\033[92mOK\033[0m'
    fail = '\033[91mFAIL\033[0m'
    crash = 'cannot be tested'

    def run_test(self, clause, test_name):
        if clause:
            print(test_name, self.ok)
            return True
        else:
            print(test_name, self.fail)
            return False

    class HidePrints:  # helper, prevents printing in function calls
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

dir_name = "test_dir"
os.chdir(dir_name + "/")

# load data for test
with open('data/index_1k.p', 'rb') as fp:
    index = pickle.load(fp)
with open('data/doc_lengths.p', 'rb') as fp:
    doc_lengths = pickle.load(fp)

checker = Checker()
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    n_tests = 4
    n_passed = 0
    try:
        module = importlib.import_module(dir_name + '.' + file[:-3])

        passed = True
        with checker.HidePrints():
            high_low_index = module.build_high_low_index(index, 5)
        if len(high_low_index) != len(index) or high_low_index['earli'][0] != {1990: 7, 3360: 6, 6190: 6, 13008: 7} or\
                len(high_low_index['earli'][1]) != 913 or high_low_index['earli'][2] != 917\
                or len(high_low_index['march'][0]) != 65 or high_low_index['march'][0][1990] != 9:
            passed = False
        if checker.run_test(passed, "build_high_low_index"): n_passed += 1


        queries = [['british', 'petroleum'],['british', 'petroleum', 'debt'],
                   ['bankamerica', 'recommend', 'make', 'deposit'], ['bahia', 'arroba']]
        answers = [{10080, 10150, 12647},{10150, 14279, 16680, 12647, 5035, 1456, 10133, 10078},
                   {5891, 4, 10117, 12292, 16648, 17164, 654, 7567, 16, 915, 15639, 540, 8349, 12449, 16161, 12195,
                    5412, 4902, 2739, 17204, 11831, 13757, 16062, 322, 2502, 15179, 9684, 5719, 7767, 5730, 16485,
                    15589, 18548, 8310, 14327, 8054, 4089},
                   {17568, 1, 11459, 14372, 11911, 14343, 16071, 13487, 14511, 14419, 13462, 13942, 14651, 15580}
                   ]
        min_docs = 3
        passed = True
        with checker.HidePrints():
            for i in range(len(queries)):
                query = Counter(queries[i])
                doc_ids = module.filter_docs(query, high_low_index, min_docs)
                if doc_ids != answers[i]:
                    passed = False
                    break
        if checker.run_test(passed, "filter_docs"): n_passed += 1

        passed = True
        with checker.HidePrints():
            query = Counter(queries[0])
            doc_ids = answers[0]
            scores = module.cosine_scoring_docs(query, doc_ids, doc_lengths, high_low_index)
            if not isclose(scores[10080], 0.082, abs_tol=1e-3) or not isclose(scores[10150], 0.201, abs_tol=1e-3):
                passed = False
        if checker.run_test(passed, "cosine_scoring_docs"): n_passed += 1

        passed = True
        with checker.HidePrints():
            query = Counter(queries[0])
            doc_ids = answers[0]
            scores = module.okapi_scoring_docs(query, doc_ids, doc_lengths, high_low_index)
            if not isclose(scores[10080], 4.142, abs_tol=1e-3) or not isclose(scores[10150], 5.402, abs_tol=1e-3):
                passed = False
        if checker.run_test(passed, "okapi_scoring_docs"): n_passed += 1

        print('%d of  %d tests passed' % (n_passed, n_tests))
    except:
        print(checker.fail, checker.crash)
