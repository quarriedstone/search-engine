import glob
import importlib
import os
import sys
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

checker = Checker()
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    n_tests = 2
    n_passed = 0
    try:
        module = importlib.import_module(dir_name + '.' + file[:-3])

        # test find_ngrams_PMI
        tokenized_text = ['on', 'saturday', ',', 'huge', 'numbers', 'of', 'venezuelans', 'took', 'to', 'the', 'streets', '—', 'most', 'of', 'them', 'to', 'show', 'their', 'support', 'for', 'opposition', 'leader', 'juan', 'guaidó', '.', 'a', 'smaller', 'contingency', 'came', 'out', 'in', 'support', 'of', 'president', 'nicolás', 'maduro', ',', 'to', 'celebrate', 'the', '20th', 'anniversary', 'of', 'the', 'rise', 'to', 'power', 'of', 'his', 'predecessor', ',', 'hugo', 'chávez', '.', 'scenes', 'from', 'the', 'two', 'sides', 'illustrate', 'the', 'deep', 'divisions', 'that', 'have', 'emerged', 'in', 'venezuela', 'in', 'recent', 'weeks', ',', 'after', 'guaidó', 'declared', 'himself', 'interim', 'president', 'and', 'maduro', 'refused', 'to', 'step', 'down', '.', 'marches', 'in', 'support', 'of', 'guaidó', ',', 'the', '35-year-old', 'head', 'of', 'the', 'opposition-controlled', 'national', 'assembly', ',', 'appear', 'to', 'have', 'attracted', 'massive', 'crowds', 'of', 'demonstrators', '—', 'holding', 'signs', 'calling', 'for', 'fair', 'elections', 'and', 'sovereign', 'democracy', '—', 'in', 'caracas', 'and', 'other', 'cities', 'around', 'the', 'country', '.', 'in', 'recent', 'years', ',', 'venezuela', 'has', 'been', 'submerged', 'in', 'a', 'state', 'of', 'political', 'and', 'humanitarian', 'turmoil', '.', 'millions', 'fled', 'the', 'country', 'as', 'migrants', 'and', 'refugees', ',', 'escaping', 'hyperinflation', 'that', 'made', 'the', 'costs', 'of', 'basic', 'goods', 'soar', '.', 'the', 'country', '’', 's', 'health', 'system', 'has', 'also', 'disintegrated', '.', 'when', 'guaidó', 'declared', 'himself', 'interim', 'president', ',', 'the', 'united', 'states', 'quickly', 'threw', 'its', 'support', 'behind', 'him', '.', 'in', 'venezuela', ',', 'maduro', 'still', 'has', 'the', 'support', 'of', 'the', 'military', '.', 'but', 'early', 'on', 'saturday', ',', 'just', 'before', 'the', 'planned', 'demonstrations', ',', 'an', 'acting', 'venezuelan', 'air', 'force', 'general', 'switched', 'sides', ',', 'throwing', 'his', 'support', 'behind', 'guaidó', 'in', 'a', 'widely', 'circulated', 'video', 'on', 'social', 'media', '.', 'in', 'the', 'short', 'clip', ',', 'he', 'says', 'that', '“', '90', 'percent', 'of', 'the', 'armed', 'forces', 'are', 'not', 'with', 'the', 'dictator.', '”', 'the', 'venezuelan', 'air', 'force', 'responded', 'on', 'twitter', ',', 'calling', 'the', 'general', 'a', 'traitor', 'and', 'claiming', 'he', '“', 'has', 'no', 'leadership', 'at', 'the', 'air', 'force.', '”', 'these', 'photos', 'offer', 'a', 'glimpse', 'into', 'what', 'the', 'demonstrations', 'look', 'like', '.']
        pmi_test_outputs = [{('interim', 'president'), ('on', 'saturday'), ('himself', 'interim'),
                             ('air', 'force'), ('declared', 'himself'), ('venezuelan', 'air')},

                            {('venezuelan', 'air', 'force'), ('guaidó', 'declared', 'himself'),
                             ('himself', 'interim', 'president'), ('declared', 'himself', 'interim')}]

        passed = True
        with checker.HidePrints():
            n2grams = module.find_ngrams_PMI(tokenized_text, 2, 6, 2)
            n3grams = module.find_ngrams_PMI(tokenized_text, 2, 12, 3)
        if n2grams != pmi_test_outputs[0] or n3grams != pmi_test_outputs[1]:
            passed = False
        if checker.run_test(passed, "pmi"): n_passed += 1

        # test build_ngram_index
        passed = True
        tokenized_docs = {
            1: ['while', 'venezuela', '’', 's', 'top', 'military', 'brass', 'has', 'come', 'out', 'in', 'support', 'of', 'mr.', 'maduro', ',', 'mr.', 'yánez', 'joined', 'a', 'growing', 'list', 'of', 'defectors', '—', 'including', 'the', 'military', 'attaché', 'of', 'the', 'venezuelan', 'embassy', 'in', 'washington', '—', 'who', 'have', 'urged', 'the', 'armed', 'forces', 'to', 'align', 'with', 'the', 'opposition', '.', 'few', 'times', 'in', 'venezuela', '’', 's', 'recent', 'history', 'has', 'so', 'much', 'appeared', 'to', 'be', 'hanging', 'on', 'a', 'protest', 'movement', '—', 'and', 'its', 'ability', 'to', 'get', 'the', 'military', ',', 'venezuela', '’', 's', 'chief', 'power', 'broker', ',', 'to', 'side', 'with', 'it', '.', '“', 'this', 'is', 'our', 'best', 'political', 'opportunity', ',', '”', 'said', 'margarita', 'lopez', 'maya', ',', 'a', 'retired', 'political', 'scientist', 'in', 'the', 'capital', ',', 'caracas', ',', 'who', 'has', 'spent', 'decades', 'studying', 'the', 'strongmen', 'of', 'the', 'country', '’', 's', 'past', 'and', 'who', 'headed', 'out', 'on', 'saturday', 'to', 'join', 'the', 'crowds', '.', '“', 'right', 'now', ',', 'it', '’', 's', 'the', 'moment', 'of', 'the', 'citizens', '.', '”'],
            2: ['francisco', 'rodríguez', ',', 'an', 'economist', 'at', 'torino', 'capital', ',', 'said', 'declared', 'himself', 'the', 'firm', 'had', 'estimated', 'turnout', 'at', 'over', '800,000', 'people', '.', 'around', 'midday', ',', 'mr.', 'guaidó', 'took', 'to', 'a', 'large', 'stage', 'before', 'his', 'supporters', 'and', 'pleaded', 'with', 'the', 'international', 'community', 'to', 'send', 'humanitarian', 'aid', 'and', 'protect', 'the', 'movement', 'he', 'has', 'fostered', '.', '“', 'the', 'next', 'days', 'will', 'be', 'decisive', '.', 'in', 'the', 'next', 'hours', ',', 'we', 'will', 'have', 'the', 'support', 'of', 'even', 'more', 'countries', ',', '”', 'he', 'said', ',', 'warning', 'that', 'the', 'world', 'was', 'watching', 'to', 'see', 'whether', 'mr.', 'maduro', '’', 's', 'security', 'forces', 'would', 'crack', 'down', '.', 'as', 'the', 'national', 'anthem', 'played', ',', 'aidé', 'de', 'ramírez', ',', 'a', '67-year-old', 'merchant', 'drenched', 'in', 'sweat', ',', 'marveled', 'that', 'there', 'were', 'no', 'security', 'forces', 'in', 'sight', ',', 'no', 'volleys', 'of', 'buckshot', ',', 'interim', 'president','no', 'clouds', 'of', 'tear', 'gas', '.', '“', 'i', 'hope', 'something', 'within', 'them', 'has', 'changed', ',', '”', 'she', 'said', '.'],
            3: ['whether', 'protests', 'alone', 'can', 'catalyze', 'a', 'shift', 'in', 'the', 'political', 'standoff', 'in', 'venezuela', 'venezuelan', 'air', 'force', 'is', 'far', 'from', 'clear', '.', 'as', 'mr.', 'maduro', 'broke', 'the', 'power', 'of', 'the', 'opposition-controlled', 'legislature', 'in', '2017', ',', 'demonstrators', 'took', 'to', 'the', 'streets', 'for', 'four', 'months', ',', 'only', 'to', 'be', 'beaten', 'back', 'in', 'clashes', 'with', 'mr.', 'maduro', '’', 's', 'security', 'forces', 'that', 'left', 'more', 'than', '100', 'people', 'dead', '.', 'but', 'this', 'time', ',', 'mr.', 'maduro', 'is', 'not', 'only', 'facing', 'a', 'challenge', 'on', 'his', 'streets', ',', 'but', 'increasing', 'unity', 'among', 'his', 'neighbors', 'in', 'the', 'region', 'that', 'his', 'rule', 'is', 'over', '.', 'mr.', 'guaidó', '’', 's', 'government', 'declared', 'himself', 'interim', 'has', 'been', 'busy', 'appointing', 'a', 'team', 'of', 'de', 'facto', 'ambassadors', 'to', 'argue', 'its', 'case', 'among', 'the', 'countries', 'that', 'have', 'recognized', 'him', '.', 'american', 'sanctions', 'on', 'venezuela', '’', 's', 'state-run', 'oil', 'company', 'could', 'topple', 'the', 'country', '’', 's', 'long-crippled', 'economy', '.', 'on', 'thursday', ',', 'the', 'european', 'parliament', 'recognized', 'mr.', 'guaidó', 'as', 'president', '.']
        }
        result = {('on', 'saturday'): [1, (1, 1)],
                  ('interim', 'president'): [1, (2, 1)],
                  ('declared', 'himself'): [2, (2, 1), (3, 1)],
                  ('venezuelan', 'air', 'force'): [1, (3, 1)],
                  ('himself', 'interim'): [1, (3, 1)],
                  ('declared', 'himself', 'interim'): [1, (3, 1)],
                  ('air', 'force'): [1, (3, 1)],
                  ('venezuelan', 'air'): [1, (3, 1)]}
        ngram_index = module.build_ngram_index(tokenized_docs, n2grams | n3grams)
        same = {k: ngram_index[k] for k in ngram_index if k in result and ngram_index[k] == result[k]}
        if len(same) != len(result):
            passed = False
        if checker.run_test(passed, "ngram_index"): n_passed += 1

        print('%d of  %d tests passed' % (n_passed, n_tests))
    except:
        print(checker.fail, checker.crash)
