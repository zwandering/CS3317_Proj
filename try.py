import numpy as np
def change(a):
    a =a + 1
    return a

a = [1,2,3,4]
b = np.asarray(a)
b = 2*b
c = np.asarray([1,1,1,1])
reasons = [1,2,3,4]
learned_clause = [1,3]
reasons_lits = np.asarray(reasons)
tmp = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)]
tmp = a/(c+0.001)
print(tmp)