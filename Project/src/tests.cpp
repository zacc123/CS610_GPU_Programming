
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/tests.h"
#include "../include/utils.h"


void run_cg_test(double *vec, unsigned int k){
    // 2 Tests, 1 Ix = vec
    // A (rand ish) * randish vec
    // set up Id matrix
    double *A = (double *)malloc(k*k*sizeof(double));
    double *AT = (double *)malloc(k*k*sizeof(double));
    double *ASpd = (double *)malloc(k*k*sizeof(double));
    
    for (unsigned int i = 0; i < k*k; i++){
        A[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }

    matrixTranspose(A, AT, k, k);
    matrixMulDense(AT, A, ASpd, k, k, k); // Now A is spd
    
    
    double *vec_A = (double *)malloc(k*sizeof(double));
    for (unsigned int i = 0; i < k; i++){
        vec_A[i] = (double)rand() / (double)RAND_MAX;
    }

    double *res_A = (double *)malloc(k*sizeof(double));
    matrixVecDense(ASpd, vec_A, res_A, k, k);

    double *x_A = (double *)calloc(k, sizeof(double)); // x0 = 0

    double *I = (double *)malloc(k*k*sizeof(double));
    memset(I, 0, sizeof(double) * k*k);
    for (unsigned int i = 0; i < k; i++){
        I[i + k*i] = 1.0;
    }
    
    double *x = (double *)malloc(k*sizeof(double));
    memset(x, 0, sizeof(double) * k);

    double atol = 1e-6;
    unsigned int max_iter = 1500;
    int iters = conjugateGradient(I, x, vec, k, atol, max_iter);
    int iters_A = conjugateGradient(ASpd, x_A, res_A, k, atol, max_iter);

    double error = 0.0;
    for (unsigned int i = 0; i < k; i++){
        error += sqrt((vec[i] - x[i])*(vec[i] - x[i]));
    }

    double errorA = 0.0;
    for (unsigned int i = 0; i < k; i++){
        errorA += sqrt((vec_A[i] - x_A[i]) * (vec_A[i] - x_A[i]));
    }
    printf("I Test: CG Finished with error: %f after %d iterations \n", error, iters);
    printf("A Test: CG Finished with error: %.12f after %d iterations \n", errorA, iters_A);


    free(I); I = NULL;
    free(x); x=NULL;
    free(A); free(AT);
    free(ASpd); free(vec_A);
    free(x_A); free(res_A);

}