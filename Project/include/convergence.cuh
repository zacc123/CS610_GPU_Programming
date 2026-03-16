#ifndef CONVERGENCE
#define CONVERGENCE

/* Test the Dense Operators, results are printed to screen */
int test_dense(int p);
int test_dense_cu(int p);
/* Test the Matrix Free Operators, results are printed to screen */
int test_mf(int p);
int test_mf_cu(int p);

#endif // CONVERGENCE