import glob
import importlib
import os
import sys


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


def load_spimi(file):
    spimi_index = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[0:-1].split(" ")  # Stop at -1 to remove the "\n" token
            line_ints = list(map(int, line[1:]))
            term = line[0]
            it = iter(line_ints)
            postings = list(zip(it, it))
            spimi_index[term] = postings
    return spimi_index


def compare_spimi(file1, file2):
    spimi1 = load_spimi(file1)
    spimi2 = load_spimi(file2)
    return spimi1 == spimi2


dir_name = "test_dir"
os.chdir(dir_name + "/")

checker = Checker()
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    n_tests = 3
    n_passed = 0

    try:
        module = importlib.import_module(dir_name + '.' + file[:-3])
        with checker.HidePrints():
            correct_spimi_path = "correct_spimi/"
            index_fname = "spimi_index.dat"
            student_spimi_path = "student_spimi/"
            reuters_data = "data/"
            n_lines = 100
            old_files = glob.glob(student_spimi_path + '*')
            for f in old_files:
                os.remove(f)  # clean folder first
            module.build_spimi_index(reuters_data, student_spimi_path, n_lines, block_size_limit_MB=0.2)

        generated_files = next(os.walk(student_spimi_path))[2]
        passed = len(generated_files) == 5
        if checker.run_test(passed, "num_files"): n_passed += 1

        file_size = os.path.getsize(student_spimi_path + "spimi_block_0.dat")
        passed = 110000 < file_size < 120000
        if checker.run_test(passed, "file_size"): n_passed += 1

        passed = compare_spimi(student_spimi_path + index_fname, correct_spimi_path + index_fname)
        if checker.run_test(passed, "spimi_index"): n_passed += 1

        print('%d of  %d tests passed' % (n_passed, n_tests))
    except:
        print(checker.fail, checker.crash)
