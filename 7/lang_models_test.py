import glob
import importlib
import os
import sys
import pickle
from collections import Counter
from math import isclose
import operator


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
with open('data/high_low_index.p', 'rb') as fp:
    high_low_index = pickle.load(fp)
with open('data/doc_lengths.p', 'rb') as fp:
    doc_lengths = pickle.load(fp)
with open('data/cat2docs.p', 'rb') as fp:
    cat2docs = pickle.load(fp)
with open('data/cat2descr.p', 'rb') as fp:
    cat2descr = pickle.load(fp)

checker = Checker()
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    n_tests = 3
    n_passed = 0
    try:
        module = importlib.import_module(dir_name + '.' + file[:-3])

        passed = True
        with checker.HidePrints():
            cat2docs_test = module.extract_categories("data/")
        if len(cat2docs_test) != 197 or set(cat2docs_test['iran']) != {43, 107, 243, 245, 386, 443} or \
                len(cat2docs_test['usa']) != 546:
            passed = False
        if checker.run_test(passed, "extract_categories"): n_passed += 1

        smoothing_add = 'additive'
        smoothing_jel = 'jelinek-mercer'

        passed = True
        with checker.HidePrints():
            query = Counter(['british', 'petroleum'])
            doc_ids = {10080, 10150, 12647}
            scores_additive = module.lm_rank_documents(query, doc_ids, doc_lengths, high_low_index, smoothing_add,
                                                       param=0.1)
            scores_jel = module.lm_rank_documents(query, doc_ids, doc_lengths, high_low_index, smoothing_jel, param=0.8)
            if not isclose(scores_additive[10080], 4.206786715743806e-06, abs_tol=1e-8) or \
                    not isclose(scores_additive[12647], 3.2814837268973182e-06, abs_tol=1e-8):
                passed = False
            if not isclose(scores_jel[10080], 0.0001597, abs_tol=1e-6) or \
                    not isclose(scores_jel[12647], 0.00107195, abs_tol=1e-6):
                passed = False
        if checker.run_test(passed, "lm_rank_documents"): n_passed += 1

        queries = [['britain', 'parliament', 'london', 'british', 'minist'], ['latin', 'america', 'countri'],
                   ['wall', 'street', 'compani', 'rate']]
        categories = [['Prime Minister Margaret Thatcher', 'lawson', 'London Stock Exchange', 'king-fahd',
                       'President Jacques Delors'],
                      ['barbados', 'President Richard Von Weizsaecker', 'President Raul Alfonsin', 'costa-rica',
                       'Finance Secretary Mario Brodersohn'],
                      ['Rupert Murdoch, publisher and chairman of News Corp Ltd', 'Australian Dollar', 'johnston',
                       'New Zealand Dollar', 'Carl Icahn']
                      ]

        passed = True
        with checker.HidePrints():
            for i, query in enumerate(queries):
                query = Counter(query)
                category_scores = module.lm_define_categories(query, cat2docs, doc_lengths, high_low_index,
                                                              smoothing_jel, param=0.9)
                top_categories = sorted(category_scores.items(), key=operator.itemgetter(1), reverse=True)[:5]
                categories_full = [x if (x not in cat2descr or cat2descr[x] == '') else cat2descr[x] for (x, y) in
                                   top_categories]
                if set(categories_full) != set(categories[i]):
                    passed = False
        if checker.run_test(passed, "lm_define_categories"): n_passed += 1

        print('%d of  %d tests passed' % (n_passed, n_tests))
    except Exception as e:
        print(e)
        print(checker.fail, checker.crash)
