
def BVE(sentence, num_vars, MAX_CLAUSE=100):
    sentence_len = len(sentence)
    print(sentence_len, num_vars)
    #c2l = {index:set(clause) for index, clause in enumerate(sentence)}
    c2l = {index: clause.copy() for index, clause in enumerate(sentence)}

    l2c = {}
    for val in range(1, num_vars+1):
        l2c[val] = set()
        l2c[-val] = set()
    for index, clause in enumerate(sentence):
        for literal in clause:
            l2c[literal].add(index)

    removed_val = []
    for val in range(1, num_vars+1):
        if len(l2c[val]) > 0 and len(l2c[val]) <= MAX_CLAUSE and \
                len(l2c[-val]) > 0 and len(l2c[-val]) <= MAX_CLAUSE:

            common = l2c[val] & l2c[-val]
            if not common:
                for clause_idx in common:
                    for literal in set(c2l[clause_idx]):
                        l2c.remove(clause_idx)
                    del c2l[clause_idx]
                if l2c[val] == 0 or l2c[-val] == 0:
                    if l2c[val] == 0 and l2c[-val] == 0:
                        removed_val.append(val)
                    continue

            R = []
            old_num_lits = 0
            old_num_lits += sum([len(c2l[c_idx]) for c_idx in l2c[val]])
            old_num_lits += sum([len(c2l[c_idx]) for c_idx in l2c[-val]])
            for clause_idx1 in l2c[val]:
                for clause_idx2 in l2c[-val]:
                    #new_clause = c2l[clause_idx1] | c2l[clause_idx2]
                    new_clause = set(c2l[clause_idx1]+c2l[clause_idx2])
                    new_clause.remove(val)
                    new_clause.remove(-val)
                    if not new_clause:
                        return [[val], [-val]], []
                    if any(-v in new_clause for v in new_clause):
                        continue
                    R.append(new_clause)
            new_num_lits = sum([len(c) for c in R])
            if old_num_lits >= new_num_lits:
                removed_val.append(val)
                for c_idx in l2c[val].copy():
                    for l in c2l[c_idx]:
                        if c_idx in l2c[l]:
                            l2c[l].remove(c_idx)
                    del c2l[c_idx]
                for c_idx in l2c[-val].copy():
                    for l in c2l[c_idx]:
                        if c_idx in l2c[l]:
                            l2c[l].remove(c_idx)
                    del c2l[c_idx]
                for i in range(len(R)):
                    # c2l[i+sentence_len]=R[i].copy()
                    c2l[i+sentence_len] = list(R[i])
                    for l in R[i]:
                        l2c[l].add(i+sentence_len)
                sentence_len += len(R)
    # print(len(c2l))
    # new_sentence = [list(clause) for clause in c2l.values()]
    # for i in range(len(sentence)):
    #     if new_sentence[i] != sentence[i]:
    #         print(i, new_sentence[i], sentence[i])
    #         break
    return [list(clause) for clause in c2l.values()], removed_val


def WhileBVE(sentence, num_vars, MAX_CLAUSE=100):
    sentence_len = len(sentence)
    print(sentence_len, num_vars)
    #c2l = {index:set(clause) for index, clause in enumerate(sentence)}
    c2l = {index: clause.copy() for index, clause in enumerate(sentence)}

    l2c = {}
    for val in range(1, num_vars+1):
        l2c[val] = set()
        l2c[-val] = set()
    for index, clause in enumerate(sentence):
        for literal in clause:
            l2c[literal].add(index)

    total_removed_val = []
    flag = True

    while flag:
        removed_val = []
        for val in range(1, num_vars+1):
            if len(l2c[val]) > 0 and len(l2c[val]) <= MAX_CLAUSE and \
                    len(l2c[-val]) > 0 and len(l2c[-val]) <= MAX_CLAUSE:

                common = l2c[val] & l2c[-val]
                if not common:
                    for clause_idx in common:
                        for literal in set(c2l[clause_idx]):
                            l2c.remove(clause_idx)
                        del c2l[clause_idx]
                    if l2c[val] == 0 or l2c[-val] == 0:
                        if l2c[val] == 0 and l2c[-val] == 0:
                            removed_val.append(val)
                        continue

                R = []
                old_num_lits = 0
                old_num_lits += sum([len(c2l[c_idx]) for c_idx in l2c[val]])
                old_num_lits += sum([len(c2l[c_idx]) for c_idx in l2c[-val]])
                for clause_idx1 in l2c[val]:
                    for clause_idx2 in l2c[-val]:
                        #new_clause = c2l[clause_idx1] | c2l[clause_idx2]
                        new_clause = set(c2l[clause_idx1]+c2l[clause_idx2])
                        new_clause.remove(val)
                        new_clause.remove(-val)
                        if not new_clause:
                            return [[val], [-val]], []
                        if any(-v in new_clause for v in new_clause):
                            continue
                        R.append(new_clause)
                new_num_lits = sum([len(c) for c in R])
                if old_num_lits >= new_num_lits:
                    removed_val.append(val)
                    for c_idx in l2c[val].copy():
                        for l in c2l[c_idx]:
                            if c_idx in l2c[l]:
                                l2c[l].remove(c_idx)
                        del c2l[c_idx]
                    for c_idx in l2c[-val].copy():
                        for l in c2l[c_idx]:
                            if c_idx in l2c[l]:
                                l2c[l].remove(c_idx)
                        del c2l[c_idx]
                    for i in range(len(R)):
                        # c2l[i+sentence_len]=R[i].copy()
                        c2l[i+sentence_len] = list(R[i])
                        for l in R[i]:
                            l2c[l].add(i+sentence_len)
                    sentence_len += len(R)
        if not removed_val:
            flag = False
        else:
            total_removed_val += removed_val

    return [list(clause) for clause in c2l.values()], total_removed_val


def Simplified_BVE(sentence, num_vars):
    sentence_len = len(sentence)
    print(sentence_len, num_vars)
    #c2l = {index:set(clause) for index, clause in enumerate(sentence)}
    c2l = {index: clause.copy() for index, clause in enumerate(sentence)}

    l2c = {}
    for val in range(1, num_vars+1):
        l2c[val] = set()
        l2c[-val] = set()
    for index, clause in enumerate(sentence):
        for literal in clause:
            l2c[literal].add(index)

    removed_val = []
    for val in range(1, num_vars+1):
        if len(l2c[val]) == 1 and len(l2c[-val]) == 1:
            removed_val.append(val)
            for idx in l2c[val]:
                idx1 = idx
            for idx in l2c[val]:
                idx2 = idx
            if idx1 == idx2:
                for literal in set(c2l[idx1]):
                    l2c[literal].remove(idx1)
                del c2l[idx1]
                continue
            new_clause = set(c2l[idx1]+c2l[idx2])
            new_clause.remove(val)
            new_clause.remove(-val)
            for literal in c2l[idx1]:
                l2c[literal].remove(idx1)
            for literal in c2l[idx2]:
                l2c[literal].remove(idx2)
            for literal in new_clause:
                l2c[literal].add(sentence_len)
            c2l[sentence_len] = new_clause.copy()
            sentence_len += 1
    return [list(clause) for clause in c2l.values()], removed_val


def postprocess(sentence, num_vars, res, removed_val):
    if not res and not removed_val:
        return None
    res = set(res)
    l2c = {}
    for val in range(1, num_vars+1):
        l2c[val] = set()
        l2c[-val] = set()
    for index, clause in enumerate(sentence):
        for literal in clause:
            l2c[literal].add(index)

    for val in reversed(removed_val):
        flag = False
        if val in res:
            res.remove(val)
        if -val in res:
            res.remove(-val)
        res.add(-val)
        for clause_idx in l2c[val]:
            if all(-l in res for l in sentence[clause_idx]):
                res.add(val)
                flag = True
                break
        if flag:
            res.remove(-val)

    return list(res)
