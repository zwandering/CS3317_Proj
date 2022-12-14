def on_assign(lit, LearntCounter, participated_v, assigned_v):
    assigned_v[lit] = LearntCounter
    participated_v[lit] = 0
    return assigned_v, participated_v

def on_unassign(literals, q_v, LearntCounter, participated_v, assigned_v, alpha):
    for lit in literals:
        interval = LearntCounter - assigned_v[lit[0]]
        if interval>0:
            r = float(participated_v[lit[0]])/float(interval)
            q_v[lit[0]] = (1.-alpha)*q_v[lit[0]] + alpha*r
    return q_v

def bcp(sentence, assignment, c2l_watch, l2c_watch, LearntCounter, participated_v, assigned_v, up_idx=0):  # NOTE: `up_idx` is for recording which assignment triggers the BCP
    """Propagate unit clauses with watched literals."""

    # For fast checking if a literal is assigned.
    assigned_lits = [a[0] for a in assignment]

    # If the assignment is empty, try BCP.
    if len(assignment) == 0:
        assert up_idx == 0

        for clause_idx, watched_lits in c2l_watch.items():
            if len(watched_lits) == 1:
                assigned_lits.append(watched_lits[0])
                assignment.append((watched_lits[0], clause_idx))
                assigned_v, participated_v = on_assign(watched_lits[0], LearntCounter, participated_v, assigned_v)

    # If it is after conflict analysis, directly assign the literal.
    elif up_idx == len(assignment):  # we use `up_idx = len(assignment)` to indicate after-conflict BCP
        neg_first_uip = sentence[-1][-1]
        assignment.append((neg_first_uip, len(sentence) - 1))
        assigned_v, participated_v = on_assign(neg_first_uip, LearntCounter, participated_v, assigned_v)

    # Propagate until no more unit clauses.
    while up_idx < len(assignment):
        lit, _ = assignment[up_idx]
        watching_clause_idxs = l2c_watch[-lit].copy()

        for clause_idx in watching_clause_idxs:
            if len(sentence[clause_idx]) == 1: return sentence[clause_idx], assigned_v, participated_v

            another_lit = c2l_watch[clause_idx][0] if c2l_watch[clause_idx][1] == -lit else c2l_watch[clause_idx][1]
            if another_lit in assigned_lits:
                continue

            is_new_lit_found = False
            for tmp_lit in sentence[clause_idx]:
                if tmp_lit != -lit and tmp_lit != another_lit and -tmp_lit not in assigned_lits:
                    c2l_watch[clause_idx].remove(-lit)
                    c2l_watch[clause_idx].append(tmp_lit)
                    l2c_watch[-lit].remove(clause_idx)
                    l2c_watch[tmp_lit].append(clause_idx)
                    is_new_lit_found = True
                    break

            if not is_new_lit_found:
                if -another_lit in assigned_lits:
                    return sentence[clause_idx], assigned_v, participated_v  # NOTE: return a clause, not the index of a clause
                else:
                    assigned_lits.append(another_lit)
                    assignment.append((another_lit, clause_idx))
                    assigned_v, participated_v = on_assign(another_lit, LearntCounter, participated_v, assigned_v)

        up_idx += 1

    return None, assigned_v, participated_v  # indicate no conflict; other return the antecedent of the conflict

def init_vsids_scores(sentence, num_vars):
    """Initialize variable scores for VSIDS."""
    scores = {}

    for lit in range(-num_vars, num_vars + 1):
        scores[lit] = 0

    for clause in sentence:
        for lit in clause:
            scores[lit] += 1

    return scores

# def decide_vsids(assignment, vsids_scores):  # NOTE: `assignment` is for filtering assigned literals
#     """Decide which variable to assign and whether to assign True or False."""
#     assigned_lit = None
#
#     # For fast checking if a literal is assigned.
#     assigned_lits = [a[0] for a in assignment]
#     unassigned_vsids_scores = vsids_scores.copy()
#     for lit in assigned_lits:
#         unassigned_vsids_scores[lit] = float("-inf")
#         unassigned_vsids_scores[-lit] = float("-inf")
#     assigned_lit = max(unassigned_vsids_scores, key=unassigned_vsids_scores.get)
#
#     return assigned_lit

def decide_q_v(assignment, q_v):  # NOTE: `assignment` is for filtering assigned literals
    """Decide which variable to assign and whether to assign True or False."""
    assigned_lit = None

    # For fast checking if a literal is assigned.
    assigned_lits = [a[0] for a in assignment]
    unassigned_q_v = q_v.copy()
    for lit in assigned_lits:
        unassigned_q_v[lit] = float("-inf")
        unassigned_q_v[-lit] = float("-inf")
    assigned_lit = max(unassigned_q_v, key=unassigned_q_v.get)

    return assigned_lit

def update_vsids_scores(vsids_scores, learned_clause, decay=0.95):
    """Update VSIDS scores."""
    for lit in learned_clause:
        vsids_scores[lit] += 1

    for lit in vsids_scores:
        vsids_scores[lit] = vsids_scores[lit] * decay

def init_watch(sentence, num_vars):
    """Initialize the watched literal data structure."""
    c2l_watch = {}  # clause -> literal
    l2c_watch = {}  # literal -> watch

    for lit in range(-num_vars, num_vars + 1):
        l2c_watch[lit] = []

    for clause_idx, clause in enumerate(sentence):  # for each clause watch first two literals
        c2l_watch[clause_idx] = []

        for lit in clause:
            if len(c2l_watch[clause_idx]) < 2:
                c2l_watch[clause_idx].append(lit)
                l2c_watch[lit].append(clause_idx)
            else:
                break

    return c2l_watch, l2c_watch

def analyze_conflict(sentence, assignment, decided_idxs, conflict_ante):  # NOTE: `sentence` is for resolution
    """Analyze the conflict with first-UIP clause learning."""
    backtrack_level, learned_clause = None, []

    # Check whether the conflict happens without making any decision.
    if len(decided_idxs) == 0:
        return -1, [], []

    # For fast checking if a literal is assigned.
    assigned_lits = [a[0] for a in assignment]

    # Find the first-UIP by repeatedly applying resolution.
    learned_clause = conflict_ante.copy()
    conflict_side = []

    while True:
        lits_at_conflict_level = assigned_lits[decided_idxs[-1]:]
        clause_lits_at_conflict_level = [-lit for lit in learned_clause if -lit in lits_at_conflict_level]

        if len(clause_lits_at_conflict_level) <= 1:
            break

        # Apply the binary resolution rule.
        is_resolved = False
        while not is_resolved:
            lit, clause_idx = assignment.pop()
            if -lit in learned_clause:
                learned_clause = list(set(learned_clause + sentence[clause_idx]))
                conflict_side = list(set(learned_clause+conflict_side))
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
        backtrack_level = next((level for level, assigned_idx in enumerate(decided_idxs) if assigned_idx > second_highest_assigned_idx), 0)

    return backtrack_level, learned_clause, conflict_side

def after_conflict_analysis(LearntCounter, learned_clause, conflict_side, participated_v, alpha):
    LearntCounter = LearntCounter + 1
    for lit in list(set(learned_clause+conflict_side)):
        participated_v[lit] = participated_v[lit] + 1
    if alpha > 0.06:
        alpha = alpha - 1e-6
    return LearntCounter, learned_clause, conflict_side, participated_v, alpha

def backtrack(assignment, decided_idxs, level, q_v, LearntCounter, participated_v, assigned_v, alpha):
    """Backtrack by deleting assigned variables."""
    literals = assignment[decided_idxs[level]:]
    q_v = on_unassign(literals, q_v, LearntCounter, participated_v, assigned_v, alpha)
    del assignment[decided_idxs[level]:]
    del decided_idxs[level:]
    return q_v

def add_learned_clause(sentence, learned_clause, c2l_watch, l2c_watch):
    """Add learned clause to the sentence and update watch."""

    # Add the learned clause to the sentence.
    sentence.append(learned_clause)

    # Update watch.
    clause_idx = len(sentence) - 1
    c2l_watch[clause_idx] = []

    # Watch the negation of the first-UIP and the literal at the backtrack level.
    # Be careful that the watched literals may be assigned.
    for lit in learned_clause[::-1]:
        if len(c2l_watch[clause_idx]) < 2:
            c2l_watch[clause_idx].append(lit)
            l2c_watch[lit].append(clause_idx)
        else:
            break

def cdcl_lrb(sentence, num_vars):
    """Run a CDCL solver for the SAT problem.

    To simplify the use of data structures, `sentence` is a list of lists where each list
    is a clause. Each clause is a list of literals, where a literal is a signed integer.
    `assignment` is also a list of literals in the order of their assignment.
    """
    # Initialize some data structures.
    alpha = 0.4
    LearntCounter = 0
    q_v={}
    assigned_v={}
    participated_v = {}
    for lit in range(-num_vars, num_vars + 1):
        q_v[lit] = 0
        assigned_v[lit] = 0
        participated_v[lit] = 0

    # vsids_scores = init_vsids_scores(sentence, num_vars)
    c2l_watch, l2c_watch = init_watch(sentence, num_vars)
    assignment, decided_idxs = [], []

    # Run BCP.
    conflict_ante, assigned_v, participated_v = \
        bcp(sentence, assignment, c2l_watch, l2c_watch, LearntCounter, participated_v, assigned_v)
    if conflict_ante is not None:
        return None  # indicate UNSAT

    # Main loop.
    while len(assignment) < num_vars:
        # Make a decision.
        assigned_lit = decide_q_v(assignment, q_v)
        decided_idxs.append(len(assignment))
        assignment.append((assigned_lit, None))
        assigned_v, participated_v = on_assign(assigned_lit, LearntCounter, participated_v, assigned_v)

        # Run BCP.
        conflict_ante, assigned_v, participated_v = \
            bcp(sentence, assignment, c2l_watch, l2c_watch, LearntCounter, participated_v, assigned_v, len(assignment) - 1)
        while conflict_ante is not None:
            # Learn conflict.
            backtrack_level, learned_clause, conflict_side = analyze_conflict(sentence, assignment, decided_idxs, conflict_ante)
            LearntCounter, learned_clause, conflict_side, participated_v, alpha = \
                after_conflict_analysis(LearntCounter, learned_clause, conflict_side, participated_v, alpha)
            add_learned_clause(sentence, learned_clause, c2l_watch, l2c_watch)

            # Update VSIDS scores.
            update_vsids_scores(vsids_scores, learned_clause)

            # Backtrack.
            if backtrack_level < 0:
                return None

            q_v = backtrack(assignment, decided_idxs, backtrack_level, q_v, LearntCounter, participated_v, assigned_v, alpha)

            # Propagate watch.
            conflict_ante = bcp(sentence, assignment, c2l_watch, l2c_watch, LearntCounter, participated_v, assigned_v, len(assignment))

    assignment = [assigned_lit for assigned_lit, _ in assignment]

    return assignment  # indicate SAT


