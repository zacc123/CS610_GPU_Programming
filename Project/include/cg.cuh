#ifndef CG_CUH
#define CG_CUH
#include "../include/matrix_free.cuh"

int conjugateGradient(double *A, double *x, double *b, int k, double atol, int max_iter);
int conjugateGradientMFCpu(double *x, double *b, int k, double atol, int max_iter, D2_mf *d2_params);
int conjugateGradient2(double *A, double *x, double *b, int k, double atol, int max_iter);
int conjugateGradientCu(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock);
int conjugateGradientCuMF(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock);
int conjugateGradientCuOpt(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock);
int conjugateGradientCuMFOpt(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock);

#endif // MATRIX_OPS_CUH