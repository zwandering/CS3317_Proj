import argparse
from cdcl_lrb2 import  CDCL_LRB
from cdcl_chb import  CDCL_CHB
from cdcl import cdcl
from utils import read_cnf
from  check import check
import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="examples/and1.cnf")
    parser.add_argument('--h', type=str, default='lrb')

    return parser.parse_args()

def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)
    start = time.time()
    # Create CDCL solver and solve it!
    heuristic = args.h

    if heuristic=='lrb':
        print('Using heuristic: lrb')
        a = CDCL_LRB(sentence, num_vars)
        res = a.run()

    if heuristic=='chb':
        print('Using heuristic: chb')
        a = CDCL_CHB(sentence, num_vars)
        res = a.run()

    if heuristic=='vsid':
        print('Using heuristic: vsid')
        res = cdcl(sentence, num_vars)

    end = time.time()
    t = end - start
    if res is None:
        print("✘ No solution found")
    else:
        if check(res, num_vars, sentence):
            print(f"✔ Successfully found a solution: {res}")
    print('总用时：'+str(t))
if __name__ == "__main__":
    args = parse_args()
    main(args)
