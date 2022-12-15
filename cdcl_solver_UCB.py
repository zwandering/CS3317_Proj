import numpy as np
import time
class CDCL_SOLVER:
    def __init__(self, sentence, num_vars):
        # Initialize class variables here
        self.sentence = sentence
        self.num_vars = num_vars
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []
        self.heuristic = self.decide_vsids  # 函数指针，指向选择的heuristic函数
        self.conflict_limit = 20            # 冲突上界，超过这个冲突次数就重启
        self.t = 0                          # 重启次数t
        self.reward = {}                    # 每种heuristic对应的reward
        self.UCB1 = {}                      # 每种heuristic对应的UCB1
        self.n = {}                         # 每种heuristic选择的次数

        # vsids需要的变量
        self.vsids_scores = {}
        self.decay=0.95
        self.init_vsids_scores()
        
        # lrb需要的变量
        self.LearntCounter = 0
        self.alpha = 0.4
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)

    def init(self):
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []

        self.vsids_scores = {}
        self.init_vsids_scores()
        
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)


    # lrb独占函数
    def on_assign(self, lit):
        self.assigned_v[lit+self.num_vars] = self.LearntCounter
        self.participated_v[lit+self.num_vars] = 0
        self.resoned_v[lit+self.num_vars] = 0

    def on_unassign(self, literals):
        assigned_lits = np.array([a[0] for a in literals])
        interval = self.LearntCounter - self.assigned_v[assigned_lits+self.num_vars]
        r = self.participated_v[assigned_lits+self.num_vars] / (interval+0.00001)
        rsr = self.resoned_v[assigned_lits + self.num_vars] / (interval+0.00001)
        self.q_v[assigned_lits + self.num_vars] = (1.0 - self.alpha) * self.q_v[assigned_lits + self.num_vars] + self.alpha * (r + rsr)

    def decide_q_v(self):  # NOTE: `assignment` is for filtering assigned literals
        """Decide which variable to assign and whether to assign True or False."""
        assigned_lit = None

        # For fast checking if a literal is assigned.
        assigned_lits = [a[0] for a in self.assignment]
        unassigned_q_v = self.q_v.copy()
        for lit in assigned_lits:
            unassigned_q_v[lit+self.num_vars] = float("-inf")
            unassigned_q_v[-lit+self.num_vars] = float("-inf")
        assigned_lit = np.argmax(unassigned_q_v) - self.num_vars

        return assigned_lit

    def after_conflict_analysis(self, learned_clause, conflict_side, reasons):
        self.LearntCounter = self.LearntCounter + 1
        tmp1 = np.concatenate((learned_clause, conflict_side))
        self.participated_v[tmp1 + self.num_vars] = self.participated_v[tmp1 + self.num_vars] + 1.
        if self.alpha > 0.06:
            self.alpha = self.alpha - 1e-6

        reasons_lits = np.asarray(reasons)
        tmp2 = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)]
        self.resoned_v[tmp2+self.num_vars] += 1.

        self.changed_lits = np.concatenate((tmp1,tmp2))

        # 使用numpy提高代码运行速度
        assigned_lits = np.array([a[0] for a in self.assignment])
        assigned_lits_neg = np.array([-lit for lit in assigned_lits])
        assigned_lits = np.union1d(assigned_lits, assigned_lits_neg)

        unassigned_lits = np.in1d(np.arange(-self.num_vars, self.num_vars + 1), assigned_lits, invert=True)
        self.q_v[unassigned_lits+self.num_vars] = 0.95 * self.q_v[unassigned_lits+self.num_vars]


    # vsids独占函数
    def init_vsids_scores(self):
        """Initialize variable scores for VSIDS."""
        for lit in range(-self.num_vars, self.num_vars + 1):
            self.vsids_scores[lit] = 0
        for clause in self.sentence:
            for lit in clause:
                self.vsids_scores[lit] += 1

    def decide_vsids(self):  # NOTE: `assignment` is for filtering assigned literals
        """Decide which variable to assign and whether to assign True or False."""
        assigned_lit = None
        # For fast checking if a literal is assigned.
        assigned_lits = [a[0] for a in self.assignment]
        unassigned_vsids_scores = self.vsids_scores.copy()
        for lit in assigned_lits:
            unassigned_vsids_scores[lit] = float("-inf")
            unassigned_vsids_scores[-lit] = float("-inf")
        assigned_lit = max(unassigned_vsids_scores, key=unassigned_vsids_scores.get)
        return assigned_lit

    def update_vsids_scores(self):
        """Update VSIDS scores."""
        for lit in self.learned_clause:
            self.vsids_scores[lit] += 1

        for lit in self.vsids_scores:
            self.vsids_scores[lit] = self.vsids_scores[lit] * self.decay


    # 共享函数
    def init_UCB1(self):
        self.UCB1[0] = 0
        self.UCB1[1] = 0

    def update_UCB1(self):
        #print("reward:", self.reward)
        print("UCB1:", self.UCB1)
        t_len = len(str(self.t))
        t = float(self.t / (10**t_len))
        t += 4
        self.UCB1[0] = self.reward[0] + np.sqrt(t/self.n[0])
        self.UCB1[1] = self.reward[1] + np.sqrt(t/self.n[1])

    def init_reward(self):
        self.reward[0] = 0
        self.reward[1] = 0

    def update_reward(self, decisions_t, decidedVars_t):
        a = 0
        if str(self.heuristic.__name__) == "decide_q_v":
            a = 0
        if str(self.heuristic.__name__) == "decide_vsids":
            a = 0
        self.reward[a] = (np.log2(decisions_t)/decidedVars_t - self.reward[a]) / self.t

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
                    if self.heuristic == self.decide_q_v:
                        self.on_assign(watched_lits[0])

        # If it is after conflict analysis, directly assign the literal.
        elif up_idx == len(self.assignment):  # we use `up_idx = len(assignment)` to indicate after-conflict BCP
            neg_first_uip = self.sentence[-1][-1]
            self.assignment.append((neg_first_uip, len(self.sentence) - 1))
            assigned_lits.add(neg_first_uip)
            if self.heuristic == self.decide_q_v:
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
                        if self.heuristic == self.decide_q_v:
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
        assignment_tmp = self.assignment.copy()
        if len(self.decided_idxs) == 0:
            return -1, [], []

        # For fast checking if a literal is assigned.
        assigned_lits = [a[0] for a in assignment_tmp]

        # Find the first-UIP by repeatedly applying resolution.
        learned_clause = set(conflict_ante.copy())
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
                lit, clause_idx = assignment_tmp.pop()
                if -lit in learned_clause:
                    reasons = list(set(reasons + self.sentence[clause_idx]))
                    learned_clause = learned_clause.union(set(self.sentence[clause_idx]))
                    conflict_side.append(lit)
                    conflict_side.append(-lit)
                    learned_clause.discard(lit)
                    learned_clause.discard(-lit)
                    # learned_clause.remove(lit)
                    # learned_clause.remove(-lit)
                    is_resolved = True

        # Order the literals of the learned clause. This is for:
        # 1) determining the backtrack level;
        # 2) watching the negation of the first-UIP and the literal at the backtrack level.
        lit_to_assigned_idx = {lit: assigned_lits.index(-lit) for lit in learned_clause}
        learned_clause = sorted(learned_clause, key=lambda lit: lit_to_assigned_idx[lit])

        # assigned_lits_set = set(assigned_lits)
        # lit_to_assigned_idx = {lit: assigned_lits.index(-lit) for lit in learned_clause if -lit in assigned_lits_set}
        # learned_clause = sorted(learned_clause, key=lambda lit: lit_to_assigned_idx[lit])

        # Decide the level to backtrack to as the second highest decision level of `learned_clause`.
        if len(learned_clause) == 1:
            backtrack_level = 0
        else:
            second_highest_assigned_idx = lit_to_assigned_idx[learned_clause[-2]]
            backtrack_level = next((level for level, assigned_idx in enumerate(self.decided_idxs) if
                                    assigned_idx > second_highest_assigned_idx), 0)

        return backtrack_level, list(learned_clause), conflict_side, reasons
    
    def backtrack(self, level):
        """Backtrack by deleting assigned variables."""
        literals = self.assignment[self.decided_idxs[level]:]
        if self.heuristic == self.decide_q_v:
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

    def __run(self):
        """Run a CDCL solver for the SAT problem.

        To simplify the use of data structures, `sentence` is a list of lists where each list
        is a clause. Each clause is a list of literals, where a literal is a signed integer.
        `assignment` is also a list of literals in the order of their assignment.
        """
        conflict_count = 0  # 这次__run的冲突计数
        decisions_t = 0     # 使用heuristic次数
        decidedVars_t = 0   # 本次__run推出的变量的数量

        # Run BCP.
        conflict_ante = self.bcp()
        if conflict_ante is not None:
            return None  # indicate UNSAT

        # Main loop.
        while len(self.assignment) < self.num_vars:
            # Make a decision.
            decisions_t += 1
            assigned_lit = self.heuristic()
            self.decided_idxs.append(len(self.assignment))
            self.assignment.append((assigned_lit, None))
            if self.heuristic == self.decide_q_v:
                self.on_assign(assigned_lit)
            
            # Run BCP.
            conflict_ante = self.bcp(len(self.assignment) - 1)
            while conflict_ante is not None:
                # Learn conflict.
                backtrack_level, learned_clause, conflict_side, reasons = self.analyze_conflict(conflict_ante)
                if self.heuristic == self.decide_q_v:
                    self.after_conflict_analysis(learned_clause, conflict_side, reasons)
                self.add_learned_clause(learned_clause)

                # Restart
                conflict_count += 1                                 # 冲突计数加一
                if conflict_count >= self.conflict_limit:           # 达到冲突上限
                    decidedVars_t = len(self.assignment)            
                    self.update_reward(decisions_t, decidedVars_t)  # 增量更新reward
                    return "Restart"

                # Backtrack.
                if backtrack_level < 0:
                    self.assignment = None
                    return None
                self.backtrack(backtrack_level)

                # Propagate watch.
                conflict_ante = self.bcp(len(self.assignment))

        self.assignment = [assigned_lit for assigned_lit, _ in self.assignment]
        return self.assignment  # indicate SAT

    def run(self):
        # 最初先运行一次lrb与vsids，更新reward
        self.init_reward()
        self.init_UCB1()
        self.n[0] = 1
        self.n[1] = 1
        self.t += 1
        self.heuristic = self.decide_q_v
        print("restart:", self.t, "with heuristic: ", self.heuristic.__name__)
        self.__run()
        self.update_UCB1()
        self.init()
        if len(self.assignment) < self.num_vars:        
            self.t += 1
            self.heuristic = self.decide_vsids
            print("restart:", self.t, "with heuristic: ", self.heuristic.__name__)
            self.__run()
            self.update_UCB1()
            self.init()

        # 接下来按照UCB1值选择heuristic并__run，直到解出
        while len(self.assignment) < self.num_vars:
            self.t += 1
            h = np.argmax(self.UCB1)
            if h == 0: self.heuristic = self.decide_vsids
            if h == 1: self.heuristic = self.decide_q_v
            print("restart:", self.t, "with heuristic: ", self.heuristic.__name__)
            res = self.__run()
            if res == None: return None
            if res == "Restart":
                self.update_UCB1()
                self.init()
        return self.assignment  # indicate SAT