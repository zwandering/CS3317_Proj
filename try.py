import numpy as np
import time
def change(a):
    a =a + 1
    return a

# a = [1,2,3,4]
# b = np.asarray(a)
# b = 2*b
# c = np.asarray([1,1,1,1])
# reasons = [1,2,3,4]
# learned_clause = [1,3]
# reasons_lits = np.asarray(reasons)
# tmp = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)]
# tmp = a/(c+0.001)
a = np.asarray([[1, 2], [3, 4],[5,6],[7,None]])
a1 = [(1, 2), (3, 4),(5,6)]
print(len(a))
a = a[:2]
del a1[2:]
print(a)
print(a1)
print(len(a),len(a1))
c = np.asarray([-4,-3,-2,-1, 0,1, 2, 3, 4])
#b = a[np.isin(a, np.arange(0,2), invert=True)]
b = np.in1d(np.arange(-4,5), a, invert=True)
b= c[np.isin(c,a,invert=True)]

# a = np.asarray([1,2])
b = np.asarray([3,4])
# a = np.concatenate((a,b))
# print(tmp)
# print(len(a))
# print(a)
# print(b)
# print(b)
# print(c[(b)+4])
# c[b] = c[b]*0.95
# print(b)
# print(c)

# start = time.time()
# for i in range(int(1e5)):
#     b = np.in1d(c, a, invert=True)
# time1 =time.time()
#
# for i in range(int(1e5)):
#     b = a[np.isin( a1,c, invert=True)]
# time2 = time.time()
#
# print(start-time1)
# print(time2-time1)