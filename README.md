# Restart

This part is the implementation for CDCL solver with restarting method.

----

Restarting: Restarting is a way of dealing with the heavy-tailed distribution of running time often found in combinatorial search. Intuitively, restarting is meant to prevent the solver from getting stuck in a part of the search space that contains no solution. The solver can often get into such situation because some incorrect assignments were committed early on, and unit resolution was unable to detect them. If the search space is large, it may take very long for these incorrect assignments to be fixed. Hence, in practice, SAT solvers usually restart after a certain number of conflicts is detected (indicating that the current search space is difficult), hoping that, with additional information, they can make better early assignments. Try incorporating a restart policy into the CDCL solver by referring to related materials, and apply classic bandit algorithms like UCB and EXP3 to switch between candidate branching heuristics after restarting for a better and more diverse exploration of the search space.

----

The newest solver is at [cdcl_solver_all_in_one.py](cdcl_solver_all_in_one.py). In [main.py](main.py), you can choose to run "res = solver.run()" or "res = solver.run_without_UCB(1)". Function solver.run() is the cdcl solver with restarting and UCB algorithm to choose new heuristic. Function solver.run_without_UCB(id) is the cdcl solver with restarting and constant heuristic, id 0 is vsids, id 1 is lrb and id 2 is chb.