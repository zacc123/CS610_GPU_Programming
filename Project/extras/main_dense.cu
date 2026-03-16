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

// void run_cg_test(double *vec, int k);

int main(void){
    // Domain Parameters
    float x0 = -100.0f;
    float xN = 100.0f;
    int ps[3] = {2, 4, 6};
    double mu = 1.0;
    int Nx;
    int Nxp;
    int Nxs[6] = {100, 200, 400, 800, 1600, 3200};
    //int Nxs[6] = {10, 10, 10, 10, 160, 320};
    double errors[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int p;



    // Matrix free Stuff
     // Add in Support for Other Operators TO DO
    int bm, bn, bs; 
    int interior;
    double d[7];
    double bd[54];
    double bsd[5];

   
    bm = 1;
    bn = 3;
    bd[0] = 1.0; bd[1] = -2.0; bd[2] = 1.0;
        
    bs = 3;
    bsd[0] = 1.5; bsd[1] = -2.0; bsd[2] = 0.5;
        
    interior = 3;
    d[0] = 1.0; d[1] = -2.0; d[2] = 1.0;
    
    for (int z = 0; z < 3; z++){

        p = ps[z];
        

        for (int j = 0; j < 6; j++ ){
            Nx = Nxs[j];
            Nxp = Nx + 1;
            // Create Domain
            float *x = create_1D_domain(x0, xN, Nx);
            double*D1, *H, *HI;
            double *D2, *SAT;
            double *BS;

            double hi = (double)Nx / ((double)(xN-x0));

            printf("Build Operators....");
            create_1D_D1_operator_dense(&D1, &H, &HI, Nx, p, x0, xN);
            create_1D_D2_operator_dense(&D2, &BS, Nx, p, x0, xN);
            
            addSAT_dir(&SAT, BS, Nx, HI, mu);
            matrixAddDenseIP(D2, SAT, Nxp, Nxp);

            double *s = (double *)calloc((Nx + 1), sizeof(double));
            double *u = (double *)calloc((Nx + 1), sizeof(double));
            double *b = (double *)calloc(Nxp, sizeof(double)); 
            double *b_spd = (double *)calloc(Nxp, sizeof(double)); 
            double *g = (double *)calloc(Nxp, sizeof(double)); 
            double *D2_t = (double *)calloc(Nxp*Nxp, sizeof(double)); 
            double *D2_spd = (double *)calloc(Nxp*Nxp, sizeof(double)); 

            D2_mf *d2_params = create_D2_mf(bm, bn, interior, bs, hi, d, bsd, bd);
            double t  = 1.0;
            data(g, x, t, Nxp);

            //memset(b, 0, Nxp*sizeof(double));
            boundary(b, SAT, g, Nxp);
            source(s, x, t, Nxp);

            vectorSubIP(b, s, Nxp);
            
            // //matrixTranspose(D2, D2_t, Nxp, Nxp);
            matrixMulDense(H, D2, D2_spd, Nxp, Nxp, Nxp);
            matrixMulScalarIP(D2_spd, -1.0, Nxp, Nxp); // make it spd
            matrixVecDense(H, b, b_spd, Nxp, Nxp);
            vectorMulScalarIP(b_spd, -1.0, Nxp);
            //  for (int i = 0; i < Nxp; i++){
            //         b[i] = (double)rand() / (double)RAND_MAX - 0.5;
            // }


            // matrixVecDense(D2_spd, b, u, Nxp, Nxp);
            // double *x, double *result, int Nxp, double h, int bm, int bn, int interior, int bs, double *D, double *BS, double *BD

            // D2matrixFreeDir(b, b_spd, Nxp, HI[0]*0.5, bm, bn, interior, bs, d, bsd, bd);


            //printf("[1] Right Before IP mul: b = %.4f\n", vectorDotProduct(b_spd, b_spd, Nxp));
            //  matrixMulDense(D2_t, D2, D2_spd, Nxp, Nxp, Nxp);
            //  matrixVecDense(D2_t, b, b_spd, Nxp, Nxp);

        
            
            //printf("[2] Right After IP mul: b = %.4f\n", vectorDotProduct(b_spd, b_spd, Nxp));
            double atol = std::sqrt(1e-9 * vectorDotProduct(b, b, Nxp));
            int max_iter = 1000;
            int threadsPerBlock = 64;
            //memset(u, 0, Nxp*sizeof(double));
            printf("Done\n");
        

            printf("Begin linear solve...");
            int iters = conjugateGradientCu(D2_spd, u, b_spd, Nxp, atol, max_iter, threadsPerBlock);
            printf("done\n");
            //printf("AFTER CG: b = %.4f\n", vectorDotProduct(b_spd, b_spd, Nxp));
            double l2_error = 0.0; 
            for (int i = 0; i<Nxp; i++){
                l2_error += (u[i] - g[i])*(u[i] - g[i]);
            }
            l2_error = l2_error * H[0];

            printf("For %d Nodes, Error is %.10f\n after %d iters\n", Nxp, l2_error, iters);
            errors[j] = std::sqrt(l2_error); // Scale it like the correct error
            // run_cg_test(u, Nxp);
            // run_cg_test_cu(u, Nxp);
            //printf("B4 FREE: b = %.4f\n", vectorDotProduct(b_spd, b_spd,Nxp));
            // Clean Up
            free_1D_domain(&x);
            free_D1_ops(&D1, &H, &HI);
            free_D2_ops(&D2, &BS);
            free(b); free(u);
            free(s); free(g);
            free(SAT);
            free(D2_t); free(D2_spd); free(b_spd);
            freeD2_mf(&d2_params);
            
        }
        break;
        printf("|Error Rate:\n");
        printf("|Error:\t|Log Error\t|\n");
        for (int i = 1; i < 6; i++){

            printf("| %.12f \t| %.12f \t| %.8f \t| %.8f\t|\n",errors[i-1], errors[i], errors[i-1]/errors[i], log2(errors[i-1]/errors[i]));
        }




    }


    
    
}
