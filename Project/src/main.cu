#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath> // Required for sqrt
#include<string.h>

#include "utils.h"
#include "sbp.h"
#include "cg.cuh"
#include "matrix_ops.cuh"
#include "tests.cuh"
#include "matrix_free.cuh"
#include "convergence.cuh"

// void run_cg_test(double *vec, int k);

int main(void){
    // Domain Parameters
    float x0 = -1.0f;
    float xN = 1.0f;
    double mu = 1.0;
    int Nx;
    int Nxp;
    // int Nxs[6] = {1875, 3750, 7500, 15000, 30000, 60000};
    int Nxs[6] = {10, 20, 40, 80, 160, 320};

    int BLAS_SIZES[6] = {1000, 2000, 4000, 8000, 16000, 32000};
    int CG_SIZES[6] = {100, 200, 400, 800, 1600, 3200};
    int threads[5] = {32, 64, 128, 256, 1024};
    int p, res;

    p = 2;

    
    bool write = true;

    // convergence  cpu dense
    res = test_dense(p);
    res = test_dense(4);
    res = test_dense(6);

    // convergence cuda dense
    res = test_dense_cu(p);
    res = test_dense_cu(4);
    res = test_dense_cu(6);

    // convergence cpu mf
    res = test_mf(2);
    res = test_mf_cu(2);
    
    for (int i = 0; i<5; i++){
        for (int j = 0; j < 6; j++ ){
            Nx = BLAS_SIZES[j];
            Nxp = Nx;
            printf("Running Test %d\n", j);
            test_BLAS_ops(p, Nxp, threads[i], write);
            test_MF_ops(Nxp, threads[i], write);
            test_cg_dense(CG_SIZES[j], threads[i], write);
            test_cg_mf(BLAS_SIZES[j]*50, threads[i], write);
            write = false;
            
            }

            // // enable for 
            // Nx = 16000;
            // Nxp = Nx;
            // test_BLAS_ops(p, Nxp, threadsPerBlock, write);
            // test_MF_ops(Nxp, threadsPerBlock, write);
        
    }
}
