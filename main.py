import argparse
from cdcl_solver_restart_LDB_MAB import CDCL_SOLVER
#from cdcl_solver_all_in_one import CDCL_SOLVER
#from cdcl_solver_restart import CDCL_SOLVER
#from cdcl_solver_UCB import CDCL_SOLVER
#from cdcl_solver import CDCL_SOLVER
from utils import read_cnf
from  check import check
import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="examples/and1.cnf")
    parser.add_argument("-d", "--decide", type=int, default=0)  # 0->vsids, 1->lrb, 2->chb
    parser.add_argument("-r", "--restart", type=int, default=1) # 0->no restart, 1->restart
    parser.add_argument("-m", "--MAB", type=int, default=1)     # 0->no MAB rechoosing, 1->MOSS, 2->UCB1
    return parser.parse_args()

def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)
    start = time.time()
    # Create CDCL solver and solve it!
    
    solver = CDCL_SOLVER(sentence, num_vars, args.restart, args.MAB, args.decide)
    res = solver.run()

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
