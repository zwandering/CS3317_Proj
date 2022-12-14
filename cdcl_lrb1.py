import numpy as np
import time
class CDCL_LRB:
    def __init__(self, sentence, num_vars):
        # Initialize class variables here
        self.LearntCounter = 0
        self.alpha = 0.4
        self.sentence = sentence
        self.num_vars = num_vars
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []

    def on_assign(self, lit):
        self.assigned_v[lit+self.num_vars] = self.LearntCounter
        self.participated_v[lit+self.num_vars] = 0
        self.resoned_v[lit+self.num_vars] = 0

    def on_unassign(self, literals):

        assigned_lits = np.array([a[0] for a in literals])
        # print(self.changed_lits)
        # print(assigned_lits)
        interval = self.LearntCounter - self.assigned_v[assigned_lits+self.num_vars]
        # print(interval)
        r = self.participated_v[assigned_lits+self.num_vars] / (interval+0.00001)
        rsr = self.resoned_v[assigned_lits + self.num_vars] / (interval+0.00001)
        self.q_v[assigned_lits + self.num_vars] = (1.0 - self.alpha) * self.q_v[assigned_lits + self.num_vars] + self.alpha * (r + rsr)

        # for lit in literals:
        #     interval = self.LearntCounter - self.assigned_v[lit[0]]
        #     if interval > 0:
        #         r = float(self.participated_v[lit[0]]) / float(interval)
        #         rsr = self.resoned_v[lit[0]+self.num_vars]/float(interval)
        #         self.q_v[lit[0]+self.num_vars] = (1.0 - self.alpha) * self.q_v[lit[0]+self.num_vars] + self.alpha * (r+rsr)

    def bcp(self, up_idx=0):
        """Propagate unit clauses with watched literals."""

        # For fast checking if a literal is assigned.
        assigned_lits = set([a[0] for a in self.assignment])

        # If the assignment is empty, try BCP.
        if len(self.assignment) == 0:
            assert up_idx == 0

            for clause_idx, watched_lits in self.c2l_watch.items():
                if len(watched_lits) == 1 and watched_lits[0] not in assigned_lits:
                    assigned_lits.add(watched_lits[0])
                    self.assignment.append((watched_lits[0], clause_idx))
                    self.on_assign(watched_lits[0])

        # If it is after conflict analysis, directly assign the literal.
        elif up_idx == len(self.assignment):  # we use `up_idx = len(assignment)` to indicate after-conflict BCP
            neg_first_uip = self.sentence[-1][-1]
            self.assignment.append((neg_first_uip, len(self.sentence) - 1))
            assigned_lits.add(neg_first_uip)
            self.on_assign(neg_first_uip, )

        # Propagate until no more unit clauses.
        while up_idx < len(self.assignment):
            lit, _ = self.assignment[up_idx]
            watching_clause_idxs = self.l2c_watch[-lit].copy()

            for clause_idx in watching_clause_idxs:
                if len(self.sentence[clause_idx]) == 1: return self.sentence[clause_idx]

                another_lit = self.c2l_watch[clause_idx][0] if self.c2l_watch[clause_idx][1] == -lit else self.c2l_watch[clause_idx][1]
                if another_lit in assigned_lits:
                    continue

                is_new_lit_found = False
                for tmp_lit in self.sentence[clause_idx]:
                    if tmp_lit != -lit and tmp_lit != another_lit and -tmp_lit not in assigned_lits:
                        self.c2l_watch[clause_idx].remove(-lit)
                        self.c2l_watch[clause_idx].append(tmp_lit)
                        self.l2c_watch[-lit].remove(clause_idx)
                        self.l2c_watch[tmp_lit].append(clause_idx)
                        is_new_lit_found = True
                        break

                if not is_new_lit_found:
                    if -another_lit in assigned_lits:
                        return self.sentence[clause_idx]  # NOTE: return a clause, not the index of a clause
                    else:
                        assigned_lits.add(another_lit)
                        self.assignment.append((another_lit, clause_idx))
                        self.on_assign(another_lit)

            up_idx += 1

        return None  # indicate no conflict; other return the antecedent of the conflict

    def decide_q_v(self):  # NOTE: `assignment` is for filtering assigned literals
        """Decide which variable to assign and whether to assign True or False."""
        assigned_lit = None

        # For fast checking if a literal is assigned.
        # for value in self.q_v:
        #     if value != 0: print(value)
        assigned_lits = [a[0] for a in self.assignment]
        unassigned_q_v = self.q_v.copy()
        for lit in assigned_lits:
            unassigned_q_v[lit+self.num_vars] = float("-inf")
            unassigned_q_v[-lit+self.num_vars] = float("-inf")
        assigned_lit = np.argmax(unassigned_q_v) - self.num_vars
        # print(assigned_lit, unassigned_q_v[assigned_lit+self.num_vars])
        # assigned_lit = max(unassigned_q_v, key=unassigned_q_v.get)

        return assigned_lit

    # def update_vsids_scores(vsids_scores, learned_clause, decay=0.95):
    #     """Update VSIDS scores."""
    #     for lit in learned_clause:
    #         vsids_scores[lit] += 1
    #
    #     for lit in vsids_scores:
    #         vsids_scores[lit] = vsids_scores[lit] * decay

    def init_watch(self):
        """Initialize the watched literal data structure."""
        c2l_watch = {}  # clause -> literal
        l2c_watch = {}  # literal -> watch

        for lit in range(-self.num_vars, self.num_vars + 1):
            l2c_watch[lit] = []

        for clause_idx, clause in enumerate(self.sentence):  # for each clause watch first two literals
            c2l_watch[clause_idx] = []

            for lit in clause:
                if len(c2l_watch[clause_idx]) < 2:
                    c2l_watch[clause_idx].append(lit)
                    l2c_watch[lit].append(clause_idx)
                else:
                    break

        return c2l_watch, l2c_watch

    def analyze_conflict(self, conflict_ante):  # NOTE: `sentence` is for resolution
        """Analyze the conflict with first-UIP clause learning."""
        backtrack_level, learned_clause = None, []

        # Check whether the conflict happens without making any decision.
        assignment_tmp = self.assignment.copy()
        if len(self.decided_idxs) == 0:
            return -1, [], []

        # For fast checking if a literal is assigned.
        assigned_lits = [a[0] for a in assignment_tmp]

        # Find the first-UIP by repeatedly applying resolution.
        learned_clause = set(conflict_ante.copy())
        conflict_side = []
        reasons = set()

        while True:
            lits_at_conflict_level = assigned_lits[self.decided_idxs[-1]:]
            clause_lits_at_conflict_level = [-lit for lit in learned_clause if -lit in lits_at_conflict_level]

            if len(clause_lits_at_conflict_level) <= 1:
                break

            # Apply the binary resolution rule.
            is_resolved = False
            while not is_resolved:
                lit, clause_idx = assignment_tmp.pop()
                if -lit in learned_clause:
                    reasons = reasons.union(set(self.sentence[clause_idx]))
                    learned_clause = set(list(learned_clause) + self.sentence[clause_idx])
                    conflict_side.append(lit)
                    conflict_side.append(-lit)
                    learned_clause.remove(lit)
                    learned_clause.remove(-lit)
                    is_resolved = True

        # Order the literals of the learned clause. This is for:
        # 1) determining the backtrack level;
        # 2) watching the negation of the first-UIP and the literal at the backtrack level.
        lit_to_assigned_idx = {lit: assigned_lits.index(-lit) for lit in learned_clause}
        learned_clause = sorted(learned_clause, key=lambda lit: lit_to_assigned_idx[lit])

        # Decide the level to backtrack to as the second highest decision level of `learned_clause`.
        if len(learned_clause) == 1:
            backtrack_level = 0
        else:
            second_highest_assigned_idx = lit_to_assigned_idx[learned_clause[-2]]
            backtrack_level = next((level for level, assigned_idx in enumerate(self.decided_idxs) if
                                    assigned_idx > second_highest_assigned_idx), 0)

        return backtrack_level, list(learned_clause), conflict_side, list(reasons)

    def after_conflict_analysis(self, learned_clause, conflict_side, reasons):
        # print(learned_clause)
        self.LearntCounter = self.LearntCounter + 1
        # print(learned_clause)
        tmp1 = np.concatenate((learned_clause, conflict_side))
        # print(tmp1)
        self.participated_v[tmp1 + self.num_vars] = self.participated_v[tmp1 + self.num_vars] + 1.
        # for lit in set(learned_clause + conflict_side):
        #     self.participated_v[lit+self.num_vars] = self.participated_v[lit+self.num_vars] + 1
        if self.alpha > 0.06:
            self.alpha = self.alpha - 1e-6

        reasons_lits = np.asarray(reasons)
        tmp2 = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)]
        self.resoned_v[tmp2+self.num_vars] += 1.

        self.changed_lits = np.concatenate((tmp1,tmp2))
        # for lit in tmp:
        #     self.resoned_v[lit] = self.resoned_v[lit] + 1

        # reasons_lits = []
        # for clause_idx in reasons:
        #     reasons_lits.extend(self.sentence[clause_idx])
        # reasons_lits = list(set(reasons_lits))
        # tmp = reasons_lits.copy()
        # for lit in learned_clause:
        #     if lit in tmp:
        #         tmp.remove(lit)
        # for lit in tmp:
        #     self.resoned_v[lit] = self.resoned_v[lit] + 1

        # 使用numpy提高代码运行速度
        assigned_lits = np.array([a[0] for a in self.assignment])
        assigned_lits_neg = np.array([-lit for lit in assigned_lits])
        assigned_lits = np.union1d(assigned_lits, assigned_lits_neg)

        unassigned_lits = np.in1d(np.arange(-self.num_vars, self.num_vars + 1), assigned_lits, invert=True)
        self.q_v[unassigned_lits+self.num_vars] = 0.95 * self.q_v[unassigned_lits+self.num_vars]
        # for lit in unassigned_lits:
        #     self.q_v[lit] = 0.95 * self.q_v[lit]

        # assigned_lits = [a[0] for a in self.assignment]
        # assigned_lits_neg = [-lit for lit in assigned_lits]
        # assigned_lits.extend(assigned_lits_neg)
        # for lit in range(-self.num_vars, self.num_vars+1):
        #     if lit not in assigned_lits:
        #         self.q_v[lit] = 0.95 * self.q_v[lit]

    def backtrack(self, level):
        """Backtrack by deleting assigned variables."""
        literals = self.assignment[self.decided_idxs[level]:]
        self.on_unassign(literals)
        del self.assignment[self.decided_idxs[level]:]
        del self.decided_idxs[level:]

    def add_learned_clause(self, learned_clause):
        """Add learned clause to the sentence and update watch."""

        # Add the learned clause to the sentence.
        self.sentence.append(learned_clause)

        # Update watch.
        clause_idx = len(self.sentence) - 1
        self.c2l_watch[clause_idx] = []

        # Watch the negation of the first-UIP and the literal at the backtrack level.
        # Be careful that the watched literals may be assigned.
        for lit in learned_clause[::-1]:
            if len(self.c2l_watch[clause_idx]) < 2:
                self.c2l_watch[clause_idx].append(lit)
                self.l2c_watch[lit].append(clause_idx)
            else:
                break

    def run(self):
        """Run a CDCL solver for the SAT problem.

        To simplify the use of data structures, `sentence` is a list of lists where each list
        is a clause. Each clause is a list of literals, where a literal is a signed integer.
        `assignment` is also a list of literals in the order of their assignment.
        """
        # start = time.time()
        # Run BCP.
        conflict_ante = self.bcp()
        if conflict_ante is not None:
            return None  # indicate UNSAT

        # Main loop.
        while len(self.assignment) < self.num_vars:
            # Make a decision.
            assigned_lit = self.decide_q_v()
            self.decided_idxs.append(len(self.assignment))
            self.assignment.append((assigned_lit, None))
            self.on_assign(assigned_lit)

            # print('time1: '+str(time1-start))
            # Run BCP.
            conflict_ante = self.bcp(len(self.assignment) - 1)
            while conflict_ante is not None:
                # Learn conflict.
                # time1 = time.time()
                backtrack_level, learned_clause, conflict_side, reasons = self.analyze_conflict(conflict_ante)
                # time2 = time.time()
                # print('analyze_conflict time: ' + str(time2 - time1))
                self.after_conflict_analysis(learned_clause, conflict_side, reasons)
                # time3 = time.time()
                # print('after_conflict_analysis time: ' + str(time3 - time2))
                self.add_learned_clause(learned_clause)

                # # Update VSIDS scores.
                # update_vsids_scores(vsids_scores, learned_clause)

                # Backtrack.
                if backtrack_level < 0:
                    return None
                # time4 = time.time()
                self.backtrack(backtrack_level)
                # time5 =time.time()
                # print('backtrack_time: '+ str(time5-time4))
                # print('\n')

                # Propagate watch.
                conflict_ante = self.bcp(len(self.assignment))

        self.assignment = [assigned_lit for assigned_lit, _ in self.assignment]

        return self.assignment  # indicate SAT
