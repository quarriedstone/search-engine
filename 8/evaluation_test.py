import glob
import importlib
import os
import sys
import pickle
import math


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
with open('data/top_k_cosine.p', 'rb') as fp:
    top_k_results = pickle.load(fp)


checker = Checker()
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    n_tests = 3
    n_passed = 0
    try:
        module = importlib.import_module(dir_name + '.' + file[:-3])
        with checker.HidePrints():
            d, q, relevance = module.read_cranfield('data/')

        passed = True
        with checker.HidePrints():
            eleven_p_interpol_avg = module.eleven_points_interpolated_avg(top_k_results, relevance, plot=True)
        true_avg = [0.6830247914282398, 0.6799390775770864, 0.5982460057329932, 0.5394671535592601,
                    0.4734442712863404, 0.4640772734715494, 0.4576637510821757, 0.4337166300264956,
                    0.3844832127632487, 0.3584430822315977, 0.3446259453767194]
        if not all(math.isclose(true_avg[k], eleven_p_interpol_avg[k], rel_tol=1e-03) for k in range(len(true_avg))):
            passed = False
        if checker.run_test(passed, "11_points_interpolated_avg"): n_passed += 1

        passed = True
        with checker.HidePrints():
            map_score = module.mean_avg_precision(top_k_results, relevance)
        if not math.isclose(map_score, 0.488666, rel_tol=1e-03):
            passed = False
        if checker.run_test(passed, "mean_avg_precision"): n_passed += 1

        passed = True
        with checker.HidePrints():
            ndcg_score = module.NDCG(top_k_results, relevance, 30)
        if not math.isclose(ndcg_score, 0.428293, rel_tol=1e-03):
            passed = False
        if checker.run_test(passed, "NDCG"): n_passed += 1

        print('%d of  %d tests passed' % (n_passed, n_tests))
    except Exception as e:
        print(e)
        print(checker.fail, checker.crash)