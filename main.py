import argparse
from cdcl_lrb1 import  CDCL_LRB
from cdcl import cdcl
from utils import read_cnf
from  check import check

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="examples/and1.cnf"
    )

    return parser.parse_args()

def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)

    # Create CDCL solver and solve it!
    cdcl = CDCL_LRB(sentence, num_vars)
    res = cdcl.run()
    # res = cdcl(sentence, num_vars)

    if res is None:
        print("✘ No solution found")
    else:
        if check(res, num_vars, sentence):
            print(f"✔ Successfully found a solution: {res}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
