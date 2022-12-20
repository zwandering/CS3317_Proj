from cdcl import cdcl
from utils import read_cnf
import argparse

def check(res, num_vars, sentence):
    import numpy as np
    assignment_lit = []
    flag = True
    for lit_tuple in res:
        assignment_lit.append(lit_tuple)
    abs_res = [abs(literal) for literal in assignment_lit]
    cnt = np.zeros(num_vars+1)
    cnt[0]=1
    for var in abs_res:
        cnt[var] +=1
    for i in range(len(cnt)):
        if cnt[i]!=1:
            print("wrong",i,cnt[i])
            # return False
    for clause in sentence:
        if not any(literal in assignment_lit for literal in clause):
            print("error",clause)
            flag = False
            # return False
    if flag:
        print("pass check!")
    return True
