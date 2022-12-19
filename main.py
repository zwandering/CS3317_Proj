import argparse
from cdcl_lrb import CDCL_LRB
from cdcl_chb import CDCL_CHB
from cdcl import cdcl
from utils import read_cnf
from check import check
from preprocess import BVE, postprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="examples/and1.cnf")
    parser.add_argument('--h', type=str, default='lrb')
    parser.add_argument('-p', '--preprocess', type=int, default=0)
    parser.add_argument('--subsumption', type=int, default=0, choices=[0, 1])

    return parser.parse_args()


def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)
    start = time.time()
    # Create CDCL solver and solve it!
    heuristic = args.h

    if args.preprocess:
        preprocess_start = time.time()
        ori_sentence = sentence.copy()
        sentence, removed_val = BVE(sentence, num_vars, args.preprocess, args.subsumption)
        preprocess_end = time.time()

    if heuristic == 'lrb':
        print('Using heuristic: lrb')
        a = CDCL_LRB(sentence, num_vars)
        res = a.run()

    if heuristic == 'chb':
        print('Using heuristic: chb')
        a = CDCL_CHB(sentence, num_vars)
        res = a.run()

    if heuristic == 'vsid':
        print('Using heuristic: vsid')
        res = cdcl(sentence, num_vars)

    if args.preprocess:
        postprocess_start = time.time()
        res = postprocess(ori_sentence, num_vars, res, removed_val)
        postprocess_end = time.time()

    end = time.time()
    t = end - start
    if res is None:
        print("✘ No solution found")
    else:
        print(f"✔ Successfully found a solution: {res}")
        check(res, num_vars, sentence)
    if args.preprocess:
        print("Preprocess time: ", preprocess_end - preprocess_start)
        print("Postprocess time: ", postprocess_end - postprocess_start)
    print('总用时：'+str(t))


if __name__ == "__main__":
    args = parse_args()
    main(args)
