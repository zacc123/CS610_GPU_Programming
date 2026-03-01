#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/sbp.h"
#include "../include/utils.h"

#define PI2 M_PI*M_PI
/*
 * Function to create 1st Derivative Operator in Dense Matrix Form
 *      INPUT: Array pointer x, start and end (x0, xN), and number of points Nx
 *          
 */
void create_1D_D1_operator_dense(double **D1, double **H, double **HI, unsigned int Nx, unsigned int p, float X0, float XN){
    // Allocate Memory for Operators
    size_t size_D1 = (Nx + 1) * (Nx + 1) * sizeof(double);
    size_t size_H = (Nx + 1) * (Nx + 1) * sizeof(double);
    *D1 = (double *)malloc(size_D1);
    *H = (double *)malloc(size_H);
    *HI = (double *)malloc(size_H);

    if (*D1 == NULL || *H == NULL || *HI == NULL){
        fprintf(stderr, "Memory Allocation in SBP Operators failed\n");
        exit(EXIT_FAILURE);
    }

    // Add in Support for Other Operators TO DO
    unsigned int bm, bn; 
    unsigned int interior;
    double d[3] = {-0.5, 0.0, 0.5};
    double bd[2] = {-1.0, 1.0};
    double bhinv[1] = {2.0};

    if (p == 2){
        bm = 1;
        bn = 2;
        interior = 3;
    }
  
    unsigned int Nxp = Nx + 1;

    // Check that dims are ok
    if (Nxp < 2*bm || Nxp < bn ){
        fprintf(stderr, "Grid not big enough to support the operator. Grid must have N >= %f\n", fmax((float) bn,(float)2*bm));
        exit(EXIT_FAILURE);
    }
    double h = (double)(XN - X0) / (double) Nx; 

    // Check that spacing makes sense
    if (h <= 0){
        fprintf(stderr, "H was negative\n");
        exit(EXIT_FAILURE);
    }

    unsigned int idx_c;
    // Fill H's with appropriate values for SBP operators
    // This is a placeholder. The actual values depend on the specific SBP scheme used.
    for (unsigned int i = 0; i < Nxp; i++){
        idx_c = i + (i * Nxp);

        (*H)[idx_c] = h;
        if (i < bm){
            (*H)[idx_c] = (*H)[idx_c] / bhinv[i];
        }

        if (i > Nxp - bm - 1){
            (*H)[idx_c] = (*H)[idx_c] / bhinv[Nxp - i];
        }
        (*HI)[idx_c] = 1.0f / (*H)[idx_c];
    }

    // D1s
    for (unsigned int i = 0; i < Nxp; i++){
        idx_c = i + (i * Nxp);

        // Handle Boundary Terms
        if (i < bm){
            for (unsigned int j = 0; j < bn; j++){
                (*D1)[j + (i * Nxp)] = bd[i*bn + j] / h;
            }
        }

        else if (i > Nxp - bm - 1){
            for (unsigned int j = 0; j < bn; j++){
                (*D1)[(Nxp + j - bn) + (i * Nxp)] = -1.0 * bd[(Nxp - i - 1)*bn + (bn - j - 1)] / h;
            }
        }
        else {
            for (unsigned int j=0; j<interior; j++){
                (*D1)[idx_c  + j - (interior / 2)] =  d[j] / h;
            }
        }
        
    }
}

void create_1D_D2_operator_dense(double **D2, double **BS, unsigned int Nx, unsigned int p, float X0, float XN){
    // Allocate Memory for Operators
    size_t size_D2 = (Nx + 1) * (Nx + 1) * sizeof(double);
    size_t size_BS = (Nx + 1) * (Nx + 1) * sizeof(double);
    *D2 = (double *)malloc(size_D2);
    *BS = (double *)malloc(size_BS);

    if (*D2 == NULL || *BS == NULL){
        fprintf(stderr, "Memory Allocation in SBP Operators failed\n");
        exit(EXIT_FAILURE);
    }

    // Add in Support for Other Operators TO DO
    unsigned int bm, bn, bs; 
    unsigned int interior;
    double d[3] = {1.0, -2.0, 1.0};
    double bd[3] = {1.0, -2.0, 1.0};
    double bhinv[1] = {2.0};
    double bsd[3] = {1.5, -2.0, 0.5};

    if (p == 2){
        bm = 1;
        bn = 3;
        interior = 3;
        bs = 3;
    }
  
    unsigned int Nxp = Nx + 1;

    // Check that dims are ok
    if (Nxp < 2*bm || Nxp < bn ){
        fprintf(stderr, "Grid not big enough to support the operator. Grid must have N >= %f\n", fmax((float) bn,(float)2*bm));
        exit(EXIT_FAILURE);
    }
    double h = ((double)XN - (double)X0) / (double) Nx; 
    double h2 = h*h;

    // Check that spacing makes sense
    if (h <= 0){
        fprintf(stderr, "H was negative\n");
        exit(EXIT_FAILURE);
    }

    unsigned int idx_c;
    // D2s
    for (unsigned int i = 0; i < Nxp; i++){
        idx_c = i + (i * Nxp);

        // Handle Boundary Terms
        if (i < bm){
            for (unsigned int j = 0; j < bn; j++){
                (*D2)[j + (i * Nxp)] = bd[i*bn + j] / h2;
            }
        }

        else if (i > Nxp - bm - 1){
            for (unsigned int j = 0; j < bn; j++){
                (*D2)[(Nxp + j - bn) + (i * Nxp)] = bd[(Nxp - i - 1)*bn + (bn - j - 1)] / h2;
            }
        }
        else {
            for (unsigned int j=0; j<interior; j++){
                (*D2)[idx_c  + j - (interior / 2)] = d[j] / h2;
            }
        }
        
    }

    // BS:
    unsigned int end_idx = Nxp*Nxp - 1;
    for (unsigned int i = 0; i < bs; i++){
        // postive side S0
        (*BS)[i] = bsd[i] / h;
        // negative side
        (*BS)[end_idx - i] = bsd[i]/h;
    }
}

void addSAT_dir(double ** SAT, double *BS, unsigned int Nx, double *HI, double mu){

    // Allocate Memory for Operators
    size_t size_SAT = (Nx + 1) * (Nx + 1) * sizeof(double);
    *SAT = (double *)malloc(size_SAT);

    double *Z = (double *)malloc(size_SAT);
    double *E0 = (double *)malloc(size_SAT);
    double *EN = (double *)malloc(size_SAT);
    double *tmp = (double *)malloc(size_SAT);
    


    memset(*SAT, (double)0.0, size_SAT);
    memset(Z, (double)0.0, size_SAT);
    memset(E0, (double)0.0, size_SAT);
    memset(EN, (double)0.0, size_SAT);
    memset(tmp, (double)0.0, size_SAT);

    double *tmp1 = (double *)malloc(size_SAT);
    memset(tmp1, (double)0.0, size_SAT);

    double *SAT1 = (double *)malloc(size_SAT);
    memset(SAT1, (double)0.0, size_SAT);

    if (*SAT == NULL || Z == NULL){
        fprintf(stderr, "Memory Allocation in SAT Operators failed\n");
        exit(EXIT_FAILURE);
    }

    // Get intermediate Matrices Set
    unsigned int idx = 0;
    double h = HI[0];

    unsigned int Nxp = Nx +1;
    double alpha1 = -13.0 * HI[0];
    double beta = 1.0;
   

    E0[0] = 1.0;
    EN[(Nxp)*(Nxp)-1] = 1.0;

    // Goal:
    // alpha1 mu HI en + beta HI (muBS)^T en
    // Term 1
    matrixMulDense(HI, EN, SAT1, Nxp, Nxp, Nxp);
    matrixMulScalarIP(SAT1, alpha1*mu, Nxp, Nxp);
    //term 2
    matrixTranspose(BS, tmp, Nxp, Nxp);
    matrixMulDense(HI, tmp, tmp1, Nxp, Nxp, Nxp);
    matrixMulScalarIP(tmp1, beta*mu, Nxp, Nxp);
    memset(tmp, 0, size_SAT);
    matrixMulDense(tmp1, EN, tmp, Nxp, Nxp, Nxp);
    //

    matrixAddDenseIP(SAT1, tmp, Nxp, Nxp);
    // SAT HAS EN TERM NOW, REPEAT FOR E0
    matrixAddDenseIP(*SAT, SAT1, Nxp, Nxp);
    
    memset(SAT1, 0, size_SAT);
    memset(tmp, 0, size_SAT);
    memset(tmp1, 0, size_SAT);

    matrixMulDense(HI, E0, SAT1, Nxp, Nxp, Nxp);
    matrixMulScalarIP(SAT1, alpha1*mu, Nxp, Nxp);
    //term 2
    matrixTranspose(BS, tmp, Nxp, Nxp);
    matrixMulDense(HI, tmp, tmp1, Nxp, Nxp, Nxp);
    matrixMulScalarIP(tmp1, beta*mu, Nxp, Nxp);
    memset(tmp, 0, size_SAT);
    matrixMulDense(tmp1, E0, tmp, Nxp, Nxp, Nxp);
    //

    matrixAddDenseIP(SAT1, tmp, Nxp, Nxp);
    // SAT HAS EN TERM NOW, REPEAT FOR E0
    matrixAddDenseIP(*SAT, SAT1, Nxp, Nxp);

    
   
    free(tmp); free(tmp1);
    free(E0); free(EN);
    free(Z);
}

void boundary(double *b, double *SAT, double *data, unsigned int Nxp){
    matrixVecDense(SAT, data, b, Nxp, Nxp); // I guess this is sort of a wrapper?/
}

void source(double *u, float *domain, double t, unsigned int Nxp){
    for (unsigned int i = 0; i < Nxp; i++){
        u[i] = 1.0 * PI2 * t * sin(M_PI * (double)domain[i]);
    }
}

void data(double *g, float *domain, double t, unsigned int Nxp){
    for (unsigned int i = 0; i < Nxp; i++){
        g[i] =  t * sin(M_PI * (double)domain[i]);
    }
}

// Clean UP D1 Ops
void free_D1_ops(double **D1, double **H, double **HI){
    free(*D1);
    *D1 = NULL;
    free(*H);
    *H = NULL;
    free(*HI);
    *HI = NULL;
}

// Clean Up D2 ops
void free_D2_ops(double **D2, double **BS){
    free(*D2);
    *D2 = NULL;
    free(*BS);
    *BS = NULL;  
}