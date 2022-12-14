import numpy as np

class CDCL_LRB:
    def __init__(self, sentence, num_vars):
        # Initialize class variables here
        self.sentence = sentence
        self.num_vars = num_vars
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []
        self.run = self.run_lrb # 函数指针，指向使用指定heuristic的run函数
        self.conflict_times = 0 # 记录冲突次数，决定是否重启
        self.conflict_limit = 5 # 冲突次数上限
        self.t = 0
        self.decisions = 0
        self.r_hat_t = {}
        self.reward_t = {}
        self.n_t_a = {}
        self.UCB_1 = {}


        self.LearntCounter = 0
        self.alpha = 0.4
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)
      
        self.vsids = {}

    # 共享的函数
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
        if len(self.decided_idxs) == 0:
            return -1, [], []

        # For fast checking if a literal is assigned.
        assigned_lits = [a[0] for a in self.assignment]

        # Find the first-UIP by repeatedly applying resolution.
        learned_clause = conflict_ante.copy()
        conflict_side = []
        reasons = []

        while True:
            lits_at_conflict_level = assigned_lits[self.decided_idxs[-1]:]
            clause_lits_at_conflict_level = [-lit for lit in learned_clause if -lit in lits_at_conflict_level]

            if len(clause_lits_at_conflict_level) <= 1:
                break

            # Apply the binary resolution rule.
            is_resolved = False
            while not is_resolved:
                lit, clause_idx = self.assignment.pop()
                if -lit in learned_clause:
                    reasons.append(clause_idx)
                    learned_clause = list(set(learned_clause + self.sentence[clause_idx]))
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

        return backtrack_level, learned_clause, conflict_side, reasons

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

    # lrb独占函数
    def on_assign(self, lit):
        self.assigned_v[lit+self.num_vars] = self.LearntCounter
        self.participated_v[lit+self.num_vars] = 0
        self.resoned_v[lit+self.num_vars] = 0

    def on_unassign(self, literals):
        assigned_lits = np.array([a[0] for a in self.assignment])
        interval = self.LearntCounter - self.assigned_v[assigned_lits++self.num_vars]
        r = self.participated_v[assigned_lits+self.num_vars] / (interval+0.01)
        rsr = self.resoned_v[assigned_lits + self.num_vars] / (interval+0.01)
        self.q_v[assigned_lits + self.num_vars] = (1.0 - self.alpha) * self.q_v[assigned_lits + self.num_vars] + self.alpha * (r + rsr)

    def decide_q_v(self):  # NOTE: `assignment` is for filtering assigned literals
        """Decide which variable to assign and whether to assign True or False."""
        assigned_lit = None

        # For fast checking if a literal is assigned.
        assigned_lits = set([a[0] for a in self.assignment])
        unassigned_q_v = self.q_v.copy()
        for lit in assigned_lits:
            unassigned_q_v[lit+self.num_vars] = float("-inf")
            unassigned_q_v[-lit+self.num_vars] = float("-inf")
        assigned_lit = np.argmax(unassigned_q_v)-self.num_vars

        return assigned_lit

    def after_conflict_analysis(self, learned_clause, conflict_side, reasons):
        #print("learned_clause: ", learned_clause)
        self.LearntCounter = self.LearntCounter + 1
        tmp = np.concatenate((learned_clause, conflict_side))
        tmp = tmp.astype(int)
        self.participated_v[tmp + self.num_vars] = self.participated_v[tmp + self.num_vars] + 1
        if self.alpha > 0.06:
            self.alpha = self.alpha - 1e-6

        reasons_lits = np.array([lit for clause_idx in reasons for lit in self.sentence[clause_idx]])
        reasons_lits = np.unique(reasons_lits)
        tmp = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)]
        tmp = tmp.astype(int)
        self.resoned_v[tmp++self.num_vars] += 1

        # 使用numpy提高代码运行速度
        assigned_lits = np.array([a[0] for a in self.assignment])
        assigned_lits_neg = np.array([-lit for lit in assigned_lits])
        assigned_lits = np.union1d(assigned_lits, assigned_lits_neg)

        unassigned_lits = np.in1d(np.arange(-self.num_vars, self.num_vars + 1), assigned_lits, invert=True)
        self.q_v[unassigned_lits+self.num_vars] = 0.95 * self.q_v[unassigned_lits+self.num_vars]

    def run_lrb(self):
        """Run a CDCL solver for the SAT problem.

        To simplify the use of data structures, `sentence` is a list of lists where each list
        is a clause. Each clause is a list of literals, where a literal is a signed integer.
        `assignment` is also a list of literals in the order of their assignment.
        """
        self.LearntCounter = 0
        self.alpha = 0.4
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)

        # Run BCP.
        conflict_ante = self.bcp()
        if conflict_ante is not None:
            return None  # indicate UNSAT

        # Main loop.
        while len(self.assignment) < self.num_vars:
            # Make a decision.
            self.decisions += 1
            assigned_lit = self.decide_q_v()
            self.decided_idxs.append(len(self.assignment))
            self.assignment.append((assigned_lit, None))
            self.on_assign(assigned_lit)

            # Run BCP.
            conflict_ante = self.bcp(len(self.assignment) - 1)
            while conflict_ante is not None:
                
                self.conflict_times += 1
                if self.conflict_times >= self.conflict_limit:
                    #print("restarting...")
                    #self.conflict_limit *= 2
                    self.reward_t[self.t] = np.log2(self.decisions) / len(self.assignment)
                    self.r_hat_t[1] += (self.reward_t[self.t] - self.r_hat_t[1])/self.t
                    self.decisions = 0
                    self.conflict_times = 0
                    self.assignment = []
                    return self.assignment

                # Learn conflict.
                backtrack_level, learned_clause, conflict_side, reasons = self.analyze_conflict(conflict_ante)
                self.after_conflict_analysis(learned_clause, conflict_side, reasons)
                self.add_learned_clause(learned_clause)

                # Backtrack.
                if backtrack_level < 0:
                    return None

                self.backtrack(backtrack_level)

                # Propagate watch.
                conflict_ante = self.bcp(len(self.assignment))

        self.assignment = [assigned_lit for assigned_lit, _ in self.assignment]

        return self.assignment  # indicate SAT

    # vsids独占函数
    def init_vsids_scores(self):
        for lit in range(-self.num_vars, self.num_vars + 1):
            self.vsids[lit] = 0
        for clause in self.sentence:
            for lit in clause:
                self.vsids[lit] += 1

    def decide_vsids(self):
        assigned_lits = [a[0] for a in self.assignment]
        unassigned_vsids_scores = self.vsids.copy()
        for lit in assigned_lits:
            unassigned_vsids_scores[lit] = float("-inf")
            unassigned_vsids_scores[-lit] = float("-inf")
        assigned_lit = max(unassigned_vsids_scores, key=unassigned_vsids_scores.get)
        return assigned_lit

    def run_vsids(self):
        """Run a CDCL solver for the SAT problem.

        To simplify the use of data structures, `sentence` is a list of lists where each list
        is a clause. Each clause is a list of literals, where a literal is a signed integer.
        `assignment` is also a list of literals in the order of their assignment.
        """
        self.init_vsids_scores()
        self.init_watch()
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.assignment, self.decided_idxs = [], []

        # Run BCP.
        conflict_ante = self.bcp()
        if conflict_ante is not None:
            return None  # indicate UNSAT

        # Main loop.
        while len(self.assignment) < self.num_vars:
            # Make a decision.
            self.decisions += 1
            assigned_lit = self.decide_vsids()
            self.decided_idxs.append(len(self.assignment))
            self.assignment.append((assigned_lit, None))

            # Run BCP.
            conflict_ante = self.bcp(len(self.assignment) - 1)
            while conflict_ante is not None:
                
                self.conflict_times += 1
                if self.conflict_times >= self.conflict_limit:
                    #print("restarting...")
                    #self.conflict_limit *= 2
                    self.reward_t[self.t] = np.log2(self.decisions) / len(self.assignment)
                    self.r_hat_t[1] += (self.reward_t[self.t] - self.r_hat_t[1])/self.t
                    self.decisions = 0
                    self.conflict_times = 0
                    self.assignment = []
                    return self.assignment

                # Learn conflict.
                backtrack_level, learned_clause, conflict_side, reasons = self.analyze_conflict(conflict_ante)
                self.add_learned_clause(learned_clause)

                # Backtrack.
                if backtrack_level < 0:
                    return None

                self.backtrack(backtrack_level)

                # Propagate watch.
                conflict_ante = self.bcp(len(self.assignment))

        self.assignment = [assigned_lit for assigned_lit, _ in self.assignment]

        return self.assignment  # indicate SAT

    def cdcl(self):
        self.n_t_a[0] = 1
        self.n_t_a[1] = 1
        self.r_hat_t[0] = 0
        self.r_hat_t[1] = 0
        self.t += 1
        self.run = self.run_lrb
        self.run()
        if len(self.assignment) < self.num_vars:
            self.t += 1
            self.run = self.run_vsids
            self.run()
        while len(self.assignment) < self.num_vars:
            self.t += 1
            self.UCB_1[0] = self.r_hat_t[0]+np.sqrt((4*np.log(self.t)) / self.n_t_a[0])
            self.UCB_1[1] = self.r_hat_t[1]+np.sqrt((4*np.log(self.t)) / self.n_t_a[1])
            print("restart")
            print("UCB1:", self.UCB_1)
            arm = np.argmax(self.UCB_1)
            if arm == 1:
                print("choose lrb")
                self.n_t_a[1] += 1
                self.run = self.run_lrb
            else:
                print("choose vsids")
                self.n_t_a[0] += 1
                self.run = self.run_vsids
            self.run()
        if self.run == self.run_vsids:
            print("finish with vsids")
            print("conflict upper bound is:", self.conflict_limit)
        else:
            print("finish with lrb")
            print("conflict upper bound is:", self.conflict_limit)
        return self.assignment