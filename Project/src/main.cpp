#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath> // Required for sqrt
#include<string.h>

#include "utils.h"
#include "sbp.h"

void run_cg_test(double *vec, unsigned int k);

int main(void){
    // Domain Parameters
    float x0 = -1.0f;
    float xN = 1.0f;
    unsigned int p = 2;
    double mu = 1.0;
    unsigned int Nx;
    unsigned int Nxp;
    unsigned int Nxs[6] = {11, 20, 40, 80, 160, 320};
    //unsigned int Nxs[6] = {10, 10, 10, 10, 160, 320};
    double errors[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (unsigned int j = 0; j < 6; j++ ){
         Nx = Nxs[j];
         Nxp = Nx + 1;
         // Create Domain
         float *x = create_1D_domain(x0, xN, Nx);
         double*D1, *H, *HI;
         double *D2, *SAT;
         double *BS;
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
        double t  = 1.0;
        data(g, x, t, Nxp);
        boundary(b, SAT, g, Nxp);
        source(s, x, t, Nxp);

        vectorSubIP(b, s, Nxp);
       
        matrixTranspose(D2, D2_t, Nxp, Nxp);
        matrixMulDense(D2_t, D2, D2_spd, Nxp, Nxp, Nxp);

        matrixVecDense(D2_t, b, b_spd, Nxp, Nxp);
        double atol = 1e-8;
        unsigned int max_iter = 10000;
        int iters = conjugateGradient(D2_spd, u, b_spd, Nxp, atol, max_iter);

        double l2_error = 0.0; 
        for (unsigned int i = 0; i<Nxp; i++){
            
            //printf("%.10f\n", u[i] - g[i]);
            l2_error += (u[i] - g[i])*(u[i] - g[i]);
         }
         l2_error = l2_error * H[0];

        printf("For %d Nodes, Error is %.10f\n after %d iters\n", Nxp, l2_error, iters);
        errors[j] = std::sqrt(l2_error);
        

        // Clean Up
        free_1D_domain(&x);
        free_D1_ops(&D1, &H, &HI);
        free_D2_ops(&D2, &BS);
        free(b); free(u);
        free(s); free(g);
        free(SAT);
        free(D2_t); free(D2_spd); free(b_spd);

      // sparse section
      // x = create_1D_domain(x0, xN, Nx);
      
      // double *D1_v; 
      // unsigned int *D1_row, *D1_col;
      // unsigned int *D1_nnz;

        
      // double *H_v;
      // unsigned int *H_row, *H_col;

      // double *HI_v;
      // unsigned int *HI_row, *HI_col;
      
      // unsigned int H_nnz = Nxp;
      // unsigned int HI_nnz = Nxp;


      // free_1D_domain(&x);
      // free(D1_v); free(D1_row); free(D1_col);
      // free(H_v); free(H_row); free(H_col);
      // free(HI_v); free(HI_row); free(HI_col);

     }

     printf("|Error Rate:\n");
     printf("|Error:\t|Log Error\t|\n");
     for (unsigned int i = 1; i < 6; i++){

        printf("| %.12f \t| %.12f \t| %.8f \t| %.8f\t|\n",errors[i-1], errors[i], errors[i-1]/errors[i], log2(errors[i-1]/errors[i]));
     }







    
    
}
