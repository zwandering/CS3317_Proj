// core/solver.cc line 591
// LBD的计算函数，统计一个clause里的所有literal一共在多少个不同的level里
// 对应的是cdcl_solver_restart_LDB.py中的mlr_Ldb
template <typename T>inline unsigned int Solver::computeLBD(const T &lits, int end) {
    int nblevels = 0;
    MYFLAG++;
#ifdef INCREMENTAL
    if(incremental) { // ----------------- INCREMENTAL MODE
      if(end==-1) end = lits.size();
      int nbDone = 0;
      for(int i=0;i<lits.size();i++) {
        if(nbDone>=end) break;
        if(isSelector(var(lits[i]))) continue;
        nbDone++;
        int l = level(var(lits[i]));
        if (permDiff[l] != MYFLAG) {
      permDiff[l] = MYFLAG;
      nblevels++;
        }
      }
    } else { // -------- DEFAULT MODE. NOT A LOT OF DIFFERENCES... BUT EASIER TO READ
#endif
    for(int i = 0; i < lits.size(); i++) {
        int l = level(var(lits[i]));
        if(permDiff[l] != MYFLAG) {
            permDiff[l] = MYFLAG;
            nblevels++;
        }
    }
#ifdef INCREMENTAL
    }
#endif
    return nblevels;
}


// core/solver.cc line 1550
// 源代码里没有独立的 afterconflict 函数，下述代码在Solver::search函数里
// 对应的是cdcl_solver_restart_LDB.py中的mlr_after_conflict和mlr_after_BCP
long double delta = nblevels - lbd_mean;
lbd_mean += delta / conflicts;
long double delta2 = nblevels - lbd_mean;
lbd_variance += delta * delta2;

long double mean = lbd_mean;
long double std_dev = sqrtl(lbd_variance / (conflicts - 1));

if (nblevels > mean + zscore * std_dev) above2std++;

if (conflictC > 3) {
    trainingSteps++;
    double feature[SGD_RESTART_FEATURES];
    feature[0] = 1;
    feature[1] = prev_lbd[0];
    feature[2] = prev_lbd[1];
    feature[3] = prev_lbd[2];
    feature[4] = prev_lbd[0] * prev_lbd[1];
    feature[5] = prev_lbd[0] * prev_lbd[2];
    feature[6] = prev_lbd[1] * prev_lbd[2];
    double predict = 0;
    for (int i = 0; i < SGD_RESTART_FEATURES; i++) {
        predict += w[i] * feature[i];
    }
    double error = predict - nblevels;
    //printf("%f %f nblevels=%d w0=%f w1=%f w2=%f w3=%f f1=%f f2=%f f3=%f    mean=%Lf\n", predict, error, nblevels, w0, w1, w2, w3, f1, f2, f3, lbd_mean);

    double alpha = 0.001;
    double epsilon = 0.00000001;
    double beta1 = 0.9;
    double beta2 = 0.999;

    double g[SGD_RESTART_FEATURES];
    for (int i = 0; i < SGD_RESTART_FEATURES; i++) {
        g[i] = error * feature[i];
    }

    for (int i = 0; i < SGD_RESTART_FEATURES; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * g[i];
    }

    for (int i = 0; i < SGD_RESTART_FEATURES; i++) {
        v[i] = beta2 * v[i] + (1 - beta2) * g[i] * g[i];
    }
                
    for (int i = 0; i < SGD_RESTART_FEATURES; i++) {
        double mhat = m[i] / (1 - pow(beta1, trainingSteps));
        double vhat = v[i] / (1 - pow(beta2, trainingSteps));
        w[i] -= alpha * mhat / (sqrt(vhat) + epsilon);
    }

    /*if (w1 < 0) w1 = 0;
    if (w1 > 1) w1 = 1;
    if (w2 < 0) w2 = 0;
    if (w2 > 1) w2 = 1;*/

                
    //printf("%f %f %f %f %f %f %f %f\n", ema_g0, ema_g1, ema_g2, ema_g3, ema_theta0, ema_theta1, ema_theta2, ema_theta3);
    //printf("%f %f %f %f\n", w0, w1, w2, w3);
}

prev_lbd[2] = prev_lbd[1];
prev_lbd[1] = prev_lbd[0];
prev_lbd[0] = nblevels;