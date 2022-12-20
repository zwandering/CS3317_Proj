# Restart

重启机制有如下两个要点：
1. 基于MAB重新选择Heuristic
   1. reward计算公式：$ r_t(a) =\frac{ \log_2 (decisions_t) }{ decidedVars_t }$
   2. 上界计算公式：
      1. UCB1: $UCB1(a)=\hat{r_t}(a)+\sqrt{\frac{4.\ln(t)}{n_t(a)}}$
      2. MOSS: $MOSS(a)=\hat{r_t}(a)+\sqrt{\frac{4}{n_t(a)}\ln(\max(\frac{t}{K.n_t(a)},1))}$
2. 基于LBD决定是否需要重启
   1. LBD计算方法：统计learned clause中所有literal所在level，learned clause中包含literal所在不同level的个数即为LBD
   2. 参考代码：[MLR source code](https://sites.google.com/a/gsd.uwaterloo.ca/maplesat/mlr)
3. - 包含LBD决策和MAB选择机制的代码在 cdcl_solver_restart_LDB_MAB.py中（代码与[MLR source code](https://sites.google.com/a/gsd.uwaterloo.ca/maplesat/mlr)参考代码之间的关系可以参考 reference_code.c）
   - 包含MAB不包含LBD的在 cdcl_solver_restart_MAB.py中
