import numpy as np
import time

class CDCL_SOLVER:
    def __init__(self, sentence, num_vars):
        # MLR
        self.mlr_alpha = 0.001
        self.mlr_epsilon = 0.00000001
        self.mlr_beta_1 = 0.9
        self.mlr_beta_2 = 0.999
        self.mlr_conflicts_since_last_restart = 0
        self.mlr_t = 0
        self.mlr_prevLbd_3 = 0
        self.mlr_prevLbd_2 = 0
        self.mlr_prevLbd_1 = 0
        self.mlr_mu = 0
        self.mlr_m2 = 0

        len_fv = len(self.mlr_feature_vector())
        self.mlr_theta = np.zeros(len_fv, dtype=float)
        self.mlr_m = np.zeros(len_fv, dtype=float)
        self.mlr_v = np.zeros(len_fv, dtype=float)


        # UCB
        self.reward = np.zeros(3,dtype=float)
        self.UCB1 = np.zeros(3,dtype=float)
        self.MOSS = np.zeros(3,dtype=float)
        self.n = np.ones(3,dtype=float)
        self.UCB = "MOSS"
        self.solver = "lrb"
        self.heuristic = self.decide_q_v
        self.t = 1

        # Initialize class variables here
        self.sentence = sentence
        self.num_vars = num_vars
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []
        self.assigned_lits = set()
        self.conflict_limit = 20
        self.conflict = 0
        self.decisions = 0

        
        # lrb & chb
        self.alpha = 0.4
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)

        # lrb
        self.LearntCounter = 0

        # chb
        self.num_conflicts = 0
        self.multi = 1.0

        # vsids
        self.vsids_scores = {}
        self.decay=0.95
        self.init_vsids_scores()

    def mlr_feature_vector(self):
        return list([1, 
        self.mlr_prevLbd_1, 
        self.mlr_prevLbd_2, 
        self.mlr_prevLbd_3, 
        self.mlr_prevLbd_1*self.mlr_prevLbd_2, 
        self.mlr_prevLbd_1*self.mlr_prevLbd_3, 
        self.mlr_prevLbd_2*self.mlr_prevLbd_3])

    def mlr_after_conflict(self, learned_clause):
        self.mlr_conflicts_since_last_restart += 1
        nextLdb = self.mlr_Ldb(learned_clause)
        delta = nextLdb - self.mlr_mu
        self.mlr_mu = self.mlr_mu + delta / self.conflict
        Delta = nextLdb - self.mlr_mu
        self.mlr_m2 = self.mlr_m2 + delta * Delta

        if self.conflict > 3:
            self.mlr_t += 1
            features = self.mlr_feature_vector()
            predict = self.mlr_theta * features
            error = predict - nextLdb
            g = error * features
            self.mlr_m = self.mlr_beta_1 * self.mlr_m + (1-self.mlr_beta_1) * g
            self.mlr_v = self.mlr_beta_2 * self.mlr_v + (1-self.mlr_beta_2) * g * g
            m_hat = self.mlr_m / (1 - self.mlr_beta_1**self.mlr_t)
            v_hat = self.mlr_v / (1 - self.mlr_beta_2**self.mlr_t)
            self.mlr_theta = self.mlr_theta - self.mlr_alpha * m_hat / (np.sqrt(v_hat)+self.mlr_epsilon)
        
        self.mlr_prevLbd_3 = self.mlr_prevLbd_2
        self.mlr_prevLbd_2 = self.mlr_prevLbd_1
        self.mlr_prevLbd_1 = nextLdb

    def mlr_after_BCP(self, is_conflict):
        if (not is_conflict) and (self.conflict > 3) and (self.mlr_conflicts_since_last_restart > 0):
            sigma = np.sqrt(self.mlr_m2/(self.conflict - 1))
            feature_vector = self.mlr_feature_vector()
            val = 0
            for i in range(len(self.mlr_theta)):
                val += self.mlr_theta[i] * feature_vector[i]
            
            if val > self.mlr_mu + 3.08 * sigma:
                return True
        return False

    def mlr_Ldb(self, learned_clause):
        def get_level(lit):
            assignment_tmp = self.assignment.copy()
            assigned_lits = [a[0] for a in assignment_tmp]
            lit_to_assigned_idx = {lit: assigned_lits.index(-lit) for lit in learned_clause}
            lit_idx = lit_to_assigned_idx[lit]
            level = next((level for level, assigned_idx in enumerate(self.decided_idxs) if
                                    assigned_idx > lit_idx), 0)
            return level
        level_list = set()
        for lit in learned_clause:
            level = get_level(lit)
            level_list.add(level)
        return len(level_list)
    
    # lrb & chb
    def decide_q_v(self):  # NOTE: `assignment` is for filtering assigned literals
        """Decide which variable to assign and whether to assign True or False."""
        assigned_lit = None

        # For fast checking if a literal is assigned.
        if len(self.assignment)==0:
            scores = self.init_vsids_scores()
            assigned_lit = max(scores)
            return assigned_lit

        assigned_lits = np.asarray(list(self.assigned_lits))
        assigned_lits = np.concatenate((assigned_lits,-assigned_lits))
        unassigned_q_v = self.q_v.copy()
        unassigned_q_v[assigned_lits+self.num_vars] = float("-inf")
        assigned_lit = np.argmax(unassigned_q_v) - self.num_vars

        return assigned_lit

    # lrb
    def on_assign(self, lit):
        self.assigned_v[lit+self.num_vars] = self.LearntCounter
        self.participated_v[lit+self.num_vars] = 0
        self.resoned_v[lit+self.num_vars] = 0

    def on_unassign(self, literals):
        lits = [a[0] for a in literals]
        assigned_lits = np.asarray(lits)
        interval = self.LearntCounter - self.assigned_v[assigned_lits+self.num_vars]
        r = self.participated_v[assigned_lits+self.num_vars] / (interval+0.00001)
        rsr = self.resoned_v[assigned_lits+self.num_vars] / (interval+0.00001)
        self.q_v[assigned_lits + self.num_vars] = (1.0 - self.alpha) * self.q_v[assigned_lits + self.num_vars] + self.alpha * (r + rsr)

    def after_conflict_analysis(self, learned_clause, conflict_side, reasons):
        self.LearntCounter = self.LearntCounter + 1
        tmp1 = np.concatenate((learned_clause, conflict_side))
        self.participated_v[tmp1 + self.num_vars] = self.participated_v[tmp1 + self.num_vars] + 1.
        if self.alpha > 0.06:
            self.alpha = self.alpha - 1e-6

        reasons_lits = np.asarray(reasons)
        tmp2 = reasons_lits[np.isin(reasons_lits, learned_clause, invert=True)] # 此处可以改为in1d可能会快一点
        self.resoned_v[tmp2+self.num_vars] += 1.

        # 使用numpy提高代码运行速度
        assigned_lits = np.asarray(list(self.assigned_lits))
        assigned_lits = np.concatenate((assigned_lits,-assigned_lits))

        unassigned_lits = np.in1d(np.arange(-self.num_vars, self.num_vars + 1), assigned_lits, invert=True)
        self.q_v[unassigned_lits] = 0.95 * self.q_v[unassigned_lits]

    # chb
    def update_q_v(self):
        plays  = np.asarray(list(self.assigned_lits))
        reward = self.multi / (self.num_conflicts - self.assigned_v[plays+self.num_vars]+1)
        self.q_v[plays + self.num_vars] = (1.-self.alpha)*self.q_v[plays+self.num_vars]+self.alpha*reward

    # vsids
    def init_vsids_scores(self):
        """Initialize variable scores for VSIDS."""
        for lit in range(-self.num_vars, self.num_vars + 1):
            self.vsids_scores[lit] = 0
        for clause in self.sentence:
            for lit in clause:
                self.vsids_scores[lit] += 1
        return self.vsids_scores

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

    def update_vsids_scores(self, learned_clause):
        """Update VSIDS scores."""
        for lit in learned_clause:
            self.vsids_scores[lit] += 1
        for lit in self.vsids_scores:
            self.vsids_scores[lit] = self.vsids_scores[lit] * self.decay

    
    # share
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

    def bcp(self, up_idx=0):
        """Propagate unit clauses with watched literals."""

        # For fast checking if a literal is assigned.
        self.assigned_lits = set([a[0] for a in self.assignment])

        # If the assignment is empty, try BCP.
        if len(self.assignment) == 0:
            assert up_idx == 0

            for clause_idx, watched_lits in self.c2l_watch.items():
                if len(watched_lits) == 1 and watched_lits[0] not in self.assigned_lits:
                    self.assigned_lits.add(watched_lits[0])
                    self.assignment.append((watched_lits[0], clause_idx))
                    if self.solver == "lrb":    # NOTE: add for lrb
                        self.on_assign(watched_lits[0])

        # If it is after conflict analysis, directly assign the literal.
        elif up_idx == len(self.assignment):  # we use `up_idx = len(assignment)` to indicate after-conflict BCP
            neg_first_uip = self.sentence[-1][-1]
            self.assignment.append((neg_first_uip, len(self.sentence) - 1))
            self.assigned_lits.add(neg_first_uip)
            if self.solver == "lrb":    # NOTE: add for lrb
                self.on_assign(neg_first_uip, )

        # Propagate until no more unit clauses.
        while up_idx < len(self.assignment):
            lit, _ = self.assignment[up_idx]
            watching_clause_idxs = self.l2c_watch[-lit].copy()

            for clause_idx in watching_clause_idxs:
                if len(self.sentence[clause_idx]) == 1: return self.sentence[clause_idx]

                another_lit = self.c2l_watch[clause_idx][0] if self.c2l_watch[clause_idx][1] == -lit else self.c2l_watch[clause_idx][1]
                if another_lit in self.assigned_lits:
                    continue

                is_new_lit_found = False
                for tmp_lit in self.sentence[clause_idx]:
                    if tmp_lit != -lit and tmp_lit != another_lit and -tmp_lit not in self.assigned_lits:
                        self.c2l_watch[clause_idx].remove(-lit)
                        self.c2l_watch[clause_idx].append(tmp_lit)
                        self.l2c_watch[-lit].remove(clause_idx)
                        self.l2c_watch[tmp_lit].append(clause_idx)
                        is_new_lit_found = True
                        break

                if not is_new_lit_found:
                    if -another_lit in self.assigned_lits:
                        return self.sentence[clause_idx]  # NOTE: return a clause, not the index of a clause
                    else:
                        self.assigned_lits.add(another_lit)
                        self.assignment.append((another_lit, clause_idx))
                        if self.solver == "lrb":    # NOTE: add for lrb
                            self.on_assign(another_lit)

            up_idx += 1

        return None  # indicate no conflict; other return the antecedent of the conflict
  
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
                    reasons.update(self.sentence[clause_idx])
                    learned_clause = learned_clause.union(set(self.sentence[clause_idx]))
                    conflict_side.append(lit)
                    conflict_side.append(-lit)
                    learned_clause.discard(lit)
                    learned_clause.discard(-lit)
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

    def backtrack(self, level):
        """Backtrack by deleting assigned variables."""
        if self.solver == "lrb":
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

    def renew(self):
        self.assigned_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.c2l_watch, self.l2c_watch = self.init_watch()
        self.assignment, self.decided_idxs = [], []
        self.assigned_lits = set()
        self.conflict_limit = 20
        self.conflict = 0
        self.decisions = 0
        # lrb & chb
        self.alpha = 0.4
        self.participated_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.q_v = np.zeros(2*self.num_vars+1, dtype = float)
        self.resoned_v = np.zeros(2*self.num_vars+1, dtype = float)
        # lrb
        self.LearntCounter = 0
        # chb
        self.num_conflicts = 0
        self.multi = 1.0
        # vsids
        self.vsids_scores = {}
        self.decay=0.95
        self.init_vsids_scores()

    def _run(self):
        """Run a CDCL solver for the SAT problem.

        To simplify the use of data structures, `sentence` is a list of lists where each list
        is a clause. Each clause is a list of literals, where a literal is a signed integer.
        `assignment` is also a list of literals in the order of their assignment.
        """
        # Run BCP.
        conflict_ante = self.bcp()
        if conflict_ante is not None:
            return None  # indicate UNSAT

        # Main loop.
        while len(self.assignment) < self.num_vars:
            # Make a decision.
            assigned_lit = self.heuristic()
            self.decided_idxs.append(len(self.assignment))
            self.assignment.append((assigned_lit, None))
            if self.solver == "lrb":
                self.on_assign(assigned_lit)
            self.decisions += 1

            # Run BCP.
            conflict_ante = self.bcp(len(self.assignment) - 1)
            if self.solver == "chb":
                if conflict_ante is None: self.multi = 0.9
                self.update_q_v()
            while conflict_ante is not None:
                # Learn conflict.
                if self.solver == "chb":
                    self.multi = 1.0
                    self.num_conflicts +=1
                    if self.alpha > 0.06: self.alpha -= 1e-6
                backtrack_level, learned_clause, conflict_side, reasons = self.analyze_conflict(conflict_ante)
                if self.solver == "lrb":
                    if backtrack_level!=-1: self.after_conflict_analysis(learned_clause, conflict_side, reasons)
                self.add_learned_clause(learned_clause)

                self.conflict += 1
                self.mlr_after_conflict(learned_clause)
                
                # Backtrack.
                if backtrack_level < 0:
                    return None
                self.backtrack(backtrack_level)

                # Propagate watch.
                conflict_ante = self.bcp(len(self.assignment))
                if self.solver == "chb":
                    self.update_q_v()

                # Restart                
                if conflict_ante is not None:
                    is_conflict = True
                else:
                    is_conflict = False

                if self.mlr_after_BCP(is_conflict):
                    reward = np.log2(self.decisions) / len(self.assignment)
                    print("assignment_len:", len(self.assignment), "sentence_len:", len(self.sentence))
                    self.renew()                    
                    return reward

        self.assignment = [assigned_lit for assigned_lit, _ in self.assignment]

        return self.assignment

    def print_restart_info(self):
        print("run with", self.solver)
        print("Reward:", self.reward)
        print("UCB1:", self.UCB1)
        print("MOSS:", self.MOSS)

    def run(self):
        self.solver = "vsids"
        self.heuristic = self.decide_vsids
        self.print_restart_info()
        reward = self._run()
        if len(self.assignment) >= self.num_vars:
            return self.assignment
        self.update_reward(0, reward)
        self.update_UCB1()
        self.update_MOSS()
        self.t += 1
        self.n[0] += 1

        self.solver = "lrb"
        self.heuristic = self.decide_q_v
        self.print_restart_info()
        reward = self._run()
        if len(self.assignment) >= self.num_vars:
            return self.assignment
        self.update_reward(1, reward)
        self.update_UCB1()
        self.update_MOSS()
        self.t += 1
        self.n[1] += 1

        self.solver = "chb"
        self.heuristic = self.decide_q_v
        self.print_restart_info()
        reward = self._run()
        if len(self.assignment) >= self.num_vars:
            return self.assignment
        self.update_reward(2, reward)
        self.update_UCB1()
        self.update_MOSS()
        self.t += 1
        self.n[2] += 1

        while len(self.assignment) < self.num_vars:
            if self.UCB == "UCB1":
                a = np.argmax(self.UCB1)
            if self.UCB == "MOSS":
                a = np.argmax(self.MOSS)
            if a == 0:
                self.solver = "vsids"
                self.heuristic = self.decide_vsids
            if a == 1:
                self.solver = "lrb"
                self.heuristic = self.decide_q_v
            if a == 2:
                self.solver = "chb"
                self.heuristic = self.decide_q_v
            self.print_restart_info()
            reward = self._run()
            if len(self.assignment) >= self.num_vars:
                return self.assignment
            self.update_reward(a, reward)
            self.update_UCB1()
            self.update_MOSS()
            self.t += 1
            self.n[a] += 1
        return self.assignment

    # UCB calculate

    def run_without_UCB(self, id):
        if id == 0:
            self.solver = "vsids"
            self.heuristic = self.decide_vsids
        if id == 1:
            self.solver = "lrb"
            self.heuristic = self.decide_q_v
        if id == 2:
            self.solver = "chb"
            self.heuristic = self.decide_q_v
        while len(self.assignment) < self.num_vars:
            self._run()
        return self.assignment

    def update_reward(self, id, reward):
        self.reward[id] += (reward - self.reward[id]) / self.t

    def update_UCB1(self):
        ln_t = np.log(self.t)
        len_ln_t = len(str(ln_t))
        ln_t /= 10**len_ln_t
        ln_t += 4
        self.UCB1[0] = self.reward[0] + np.sqrt(ln_t / self.n[0])
        self.UCB1[1] = self.reward[1] + np.sqrt(ln_t / self.n[1])
        self.UCB1[2] = self.reward[2] + np.sqrt(ln_t / self.n[2])

    def update_MOSS(self):
        n0 = self.n[0]
        n1 = self.n[1]
        n2 = self.n[2]
        n0 /= 10**len(str(n0))
        n1 /= 10**len(str(n1))
        n2 /= 10**len(str(n2))
        n0 += 3
        n1 += 3
        n2 += 3

        self.MOSS[0] = self.reward[0] + np.sqrt((4/self.n[0]) * np.log(np.max([1, (self.t / n0)])))
        self.MOSS[1] = self.reward[1] + np.sqrt((4/self.n[1]) * np.log(np.max([1, (self.t / n1)])))
        self.MOSS[2] = self.reward[2] + np.sqrt((4/self.n[2]) * np.log(np.max([1, (self.t / n2)])))