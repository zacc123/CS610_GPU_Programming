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

    // These arrays based on max num elements in p=6
    double d[7];
    double bd[54];
    double bhinv[6];

    if (p == 2){
        bm = 1;
        bn = 2;
        bd[0] = -1.0; bd[1] = 1.0;
        interior = 3;
        d[0] = -0.5; d[1] = 0; d[2] = 0.5;
        bhinv[0] = 2.0;
    
    }

    if (p == 4){
        // Set Bdry Derv
        bm = 4;
        bn = 6;
        bd[0] = -1.411764705882353;     bd[1] = 1.7352941176470589;     bd[2] =  -0.23529411764705882;   bd[3] = -0.08823529411764706; bd[4]  = 0.0;                 bd[5] =  0.0;
        bd[6] = -0.5;                   bd[7] = 0.0;                    bd[8] =   0.5;                   bd[9] =  0.0;                 bd[10] = 0.0;                 bd[11] = 0.0;
        bd[12] = 0.09302325581395349;   bd[13] = -0.686046511627907;    bd[14] =  0.0;                   bd[15] = 0.686046511627907;   bd[16] =-0.09302325581395349; bd[17] = 0.0;
        bd[18] = 0.030612244897959183;  bd[19] = 0.0;                   bd[20] = -0.6020408163265306;    bd[21] = 0.0;                 bd[22] = 0.6530612244897959;  bd[23] = -0.08163265306122448;
        interior = 5;
        d[0] = 0.08333333333333333; d[1] = -0.6666666666666666; d[2] = 0.0;  d[3] = 0.6666666666666666; d[4] = -0.08333333333333333;
       
        bhinv[0] = 2.823529411764706; bhinv[1] = 0.8135593220338984; bhinv[2] = 1.1162790697674418; bhinv[3] = 0.9795918367346939;
        
    }

    if (p == 6){
        // Set Bdry Derv
        bm = 6;
        bn = 9;
        
        // R0
        bd[0] = -1.5825335189391163;    bd[1] = 2.033378678700676;      bd[2] = -0.14151285874487368;
        bd[3] = -0.45039830657827157;   bd[4] = 0.10448806928404068;    bd[5] = 0.03657793627754379;
        bd[6] = 0.0;                    bd[7] = 0.0;                    bd[8] = 0.0;

        // R1
        bd[9] = -0.4620591956311584;    bd[10] = 0.0;                   bd[11] = 0.28725862297825056;
        bd[12] = 0.25881608737683226;   bd[13] = -0.06911206553262328;  bd[14] = -0.014903449191300218;
        bd[15] = 0.0;                   bd[16] = 0.0;                   bd[17] = 0.0;

        // R2
        bd[18] = 0.07124710472182888;   bd[19] = -0.6364510951379055;   bd[20] = 0.0;
        bd[21] = 0.6062355236091443;    bd[22] = -0.02290219027581106;  bd[23] = -0.018129342917256683;
        bd[24] = 0.0;                   bd[25] = 0.0;                   bd[26] = 0.0;

        // R3
        bd[27] = 0.1147133137989705;    bd[28] = -0.29008748438681525;  bd[29] = -0.3066811913611477;
        bd[30] = 0.0;                   bd[31] = 0.5202622850504811;    bd[32] = -0.05164226551611832;
        bd[33] = 0.013435342414629596;  bd[34] = 0.0;                   bd[35] = 0.0;

        // R4
        bd[36] = -0.0362106806565411;   bd[37] = 0.10540094493378233;   bd[38] = 0.01576433612738956;
        bd[39] = -0.707905442575988;    bd[40] = 0.0;                   bd[41] = 0.7691994139626472;
        bd[42] = -0.1645296432652025;   bd[43] = 0.01828107147391139;   bd[44] = 0.0;

        // R5
        bd[45] = -0.011398193015049775; bd[46] = 0.020437334208704083;  bd[47] = 0.011220896474665617;
        bd[48] = 0.06318369464187532;   bd[49] = -0.6916490244268136;   bd[50] = 0.0;
        bd[51] = 0.7397091390607521;    bd[52] = -0.1479418278121504;   bd[53] = 0.016437980868016712;


        interior = 7;
        d[0] = -0.016666666666666666; d[1] = 0.15; d[2] = -0.75;  
        d[3] = 0;                     d[4] = 0.75; d[5] = -0.15;
        d[6] =  0.016666666666666666;
       
        bhinv[0] = 3.1650670378782326; bhinv[1] = 0.719220844085574;    bhinv[2] = 1.5935079306528956; 
        bhinv[3] = 0.8061205448777757; bhinv[4] = 1.0968642884346833;   bhinv[5] = 0.9862788520810027;
    }

    if ((p!= 2) && (p!= 4) && (p != 6)){
        fprintf(stderr, "%u not a supported order. Code only supports p = 2,4,6\n", p);
        exit(EXIT_FAILURE);
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
            (*H)[idx_c] = (*H)[idx_c] / bhinv[Nxp - i - 1];
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
    double d[7];
    double bd[54];
    double bsd[5];

    if (p == 2){
        bm = 1;
        bn = 3;
        bd[0] = 1.0; bd[1] = -2.0; bd[2] = 1.0;
        
        bs = 3;
        bsd[0] = 1.5; bsd[1] = -2.0; bsd[2] = 0.5;
        
        interior = 3;
        d[0] = 1.0; d[1] = -2.0; d[2] = 1.0;
    }

    if (p == 4){
        // printf("Doing P = 4\n");
        bm = 4;
        bn = 6;
        bd[0] = 2.0; bd[1] = -5.0; bd[2] = 4.0; bd[3] = -1.0;   bd[4] = 0.0;  bd[5] = 0.0;
        bd[6] = 1.0; bd[7] = -2.0; bd[8] = 1.0; bd[9] =  0.0;  bd[10] = 0.0; bd[11] = 0.0;
        
        bd[12] = -0.09302325581395349; bd[13] = 1.372093023255814; bd[14] = -2.558139534883721; bd[15] =  1.372093023255814;  bd[16] = -0.09302325581395349; bd[17] = 0.0;
        bd[18] = -0.02040816326530612; bd[19] = 0.0;               bd[20] = 1.2040816326530612; bd[21] =  -2.4081632653061225;  bd[22] = 1.3061224489795917; bd[23] = -0.08163265306122448;

        bs = 4;
        bsd[0] = 1.8333333333333333; bsd[1] = -3.0; bsd[2] = 1.5; bsd[3] = -0.3333333333333333;
        
        interior = 5;
        d[0] = -0.08333333333333333; d[1] = 1.3333333333333333; d[2] = -2.5; d[3] = 1.3333333333333333; d[4] = -0.08333333333333333;
    }

    if (p==6){
        bm = 6;
        bn = 9;

        // R0
        bd[0] = 2.7882384545876375;    bd[1] = -8.024525606271522;      bd[2] = 8.215717879209711;
        bd[3] = -3.3823845458763766;   bd[4] = 0.2745256062715217;      bd[5] = 0.128428212079029;
        bd[6] = 0.0;                   bd[7] = 0.0;                     bd[8] = 0.0;

        // R1
        bd[9] = 1.0534129692832765;    bd[10] = -2.3503981797497158;    bd[11] = 1.8674630261660978;
        bd[12] = -1.0341296928327646;  bd[13] = 0.6003981797497155;     bd[14] = -0.13674630261660978;
        bd[15] = 0.0;                  bd[16] = 0.0;                    bd[17] = 0.0;

        // R2
        bd[18] = -0.64417804008361;    bd[19] = 4.137556867084717;      bd[20] = -8.108447067502766;
        bd[21] = 6.9417804008361;      bd[22] = -2.8875568670847165;    bd[23] = 0.5608447067502766;
        bd[24] = 0.0;                  bd[25] = 0.0;                    bd[26] = 0.0;

        // R3
        bd[27] = 0.21335759159047085;  bd[28] = -1.159078186228774;     bd[29] = 3.511693723953474;
        bd[30] = -4.7231448653355725;  bd[31] = 2.4896902407165515;     bd[32] = -0.3414753996392362;
        bd[33] = 0.008956894943086397; bd[34] = 0.0;                    bd[35] = 0.0;

        // R4
        bd[36] = -0.17907832931319031; bd[37] = 0.9156510515847827;     bd[38] = -1.987601032542;
        bd[39] = 3.3876475815665863;   bd[40] = -3.7198595065803395;    bd[41] = 1.7355824975667555;
        bd[42] = -0.1645296432652025;  bd[43] = 0.012187380982607592;   bd[44] = 0.0;

        // R5
        bd[45] = 0.040020014763742076; bd[46] = -0.18752235489296287;   bd[47] =0.34712677792744456;
        bd[48] = -0.41779107021909695; bd[49] = 1.560601736642238;      bd[50] = -2.68487020844273;
        bd[51] = 1.4794182781215042;   bd[52] = -0.1479418278121504;    bd[53] = 0.01095865391201114;

        bs = 5;
        bsd[0] = 2.0833333333333335; bsd[1] = -4.0; bsd[2] = 3.0; bsd[3] = -1.3333333333333333; bsd[4] = 0.25;

        interior = 7;
        d[0] = 0.011111111111111112; d[1] = -0.15; d[2] = 1.5; d[3] = -2.7222222222222223;
        d[6] = 0.011111111111111112; d[5] = -0.15; d[4] = 1.5;
    }
  

    unsigned int Nxp = Nx + 1;

    // Check that dims are ok
    if (Nxp < 2*bm || Nxp < bn ){
        fprintf(stderr, "Grid not big enough to support the operator. Grid must have N >= %f\n", fmax((float) bn,(float)2*bm));
        exit(EXIT_FAILURE);
    }

    if ((p!= 2) && (p!= 4) && (p != 6)){
        fprintf(stderr, "%u not a supported order. Code only supports p = 2,4,6\n", p);
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
    printf("h2 == %.2f\n", h2);
    for (unsigned int i = 0; i < Nxp; i++){
        idx_c = i + (i * Nxp);

        // Handle Boundary Terms
        if (i < bm){
            for (unsigned int j = 0; j < bn; j++){
                //printf("at index %u %u d2 = %.2f\n", i, j,  bd[i*bn + j] / h2);
                (*D2)[j + (i * Nxp)] = bd[i*bn + j] / h2;
            }
        }

        // else if (i > Nxp - bm - 1){
        //     for (unsigned int j = 0; j < bn; j++){
        //         (*D2)[(Nxp + j - bn) + (i * Nxp)] = bd[(Nxp - i - 1)*bn + (bn - j - 1)] / h2;
        //     }
        // }
        // else {
        //     for (unsigned int j=0; j<interior; j++){
        //         (*D2)[idx_c  + j - (interior / 2)] = d[j] / h2;
        //     }
        // }
        else if (i > Nxp - bm - 1){
            for (unsigned int j = 0; j < bn; j++){
                (*D2)[(Nxp + j - bn) + (i * Nxp)] = 1.0 * bd[(Nxp - i - 1)*bn + (bn - j - 1)] / h2;
            }
        }
        else {
            for (unsigned int j=0; j<interior; j++){
                (*D2)[idx_c  + j - (interior / 2)] =  d[j] / h2;
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




/*
 *
 * SPARSE OPERATORS
 *
 *
 *
 */

 void create_1D_D1_operator_sparse(double **D1_v, double **D1_row, double **D1_col, double *D1_nnz, double **H_v, double **H_row, double **H_col, double **HI_v, double **HI_row, double **HI_col, unsigned int Nx, unsigned int p, float X0, float XN){
    // Allocate Memory for Operators

    // Main difference from dense:
    //  - We have to get order before knowing the size 
    //  - Will want to track sizes of arrays (m x n) and num of non zeros
    unsigned int Nxp = Nx + 1;
    
    // Add in Support for Other Operators TO DO
    unsigned int bm, bn;
    unsigned int interior;

    // These arrays based on max num elements in p=6
    double d[7];
    double bd[54];
    double bhinv[6];

    if (p == 2){
        bm = 1;
        bn = 2;
        bd[0] = -1.0; bd[1] = 1.0;
        interior = 3;
        d[0] = -0.5; d[1] = 0; d[2] = 0.5;
        bhinv[0] = 2.0;
        
    }

    if (p == 4){
        // Set Bdry Derv
        bm = 4;
        bn = 6;
        bd[0] = -1.411764705882353;     bd[1] = 1.7352941176470589;     bd[2] =  -0.23529411764705882;   bd[3] = -0.08823529411764706; bd[4]  = 0.0;                 bd[5] =  0.0;
        bd[6] = -0.5;                   bd[7] = 0.0;                    bd[8] =   0.5;                   bd[9] =  0.0;                 bd[10] = 0.0;                 bd[11] = 0.0;
        bd[12] = 0.09302325581395349;   bd[13] = -0.686046511627907;    bd[14] =  0.0;                   bd[15] = 0.686046511627907;   bd[16] =-0.09302325581395349; bd[17] = 0.0;
        bd[18] = 0.030612244897959183;  bd[19] = 0.0;                   bd[20] = -0.6020408163265306;    bd[21] = 0.0;                 bd[22] = 0.6530612244897959;  bd[23] = -0.08163265306122448;
        interior = 5;
        d[0] = 0.08333333333333333; d[1] = -0.6666666666666666; d[2] = 0.0;  d[3] = 0.6666666666666666; d[4] = -0.08333333333333333;
       
        bhinv[0] = 2.823529411764706; bhinv[1] = 0.8135593220338984; bhinv[2] = 1.1162790697674418; bhinv[3] = 0.9795918367346939;
         
    }

    if (p == 6){
        // Set Bdry Derv
        bm = 6;
        bn = 9;
        
        // R0
        bd[0] = -1.5825335189391163;    bd[1] = 2.033378678700676;      bd[2] = -0.14151285874487368;
        bd[3] = -0.45039830657827157;   bd[4] = 0.10448806928404068;    bd[5] = 0.03657793627754379;
        bd[6] = 0.0;                    bd[7] = 0.0;                    bd[8] = 0.0;

        // R1
        bd[9] = -0.4620591956311584;    bd[10] = 0.0;                   bd[11] = 0.28725862297825056;
        bd[12] = 0.25881608737683226;   bd[13] = -0.06911206553262328;  bd[14] = -0.014903449191300218;
        bd[15] = 0.0;                   bd[16] = 0.0;                   bd[17] = 0.0;

        // R2
        bd[18] = 0.07124710472182888;   bd[19] = -0.6364510951379055;   bd[20] = 0.0;
        bd[21] = 0.6062355236091443;    bd[22] = -0.02290219027581106;  bd[23] = -0.018129342917256683;
        bd[24] = 0.0;                   bd[25] = 0.0;                   bd[26] = 0.0;

        // R3
        bd[27] = 0.1147133137989705;    bd[28] = -0.29008748438681525;  bd[29] = -0.3066811913611477;
        bd[30] = 0.0;                   bd[31] = 0.5202622850504811;    bd[32] = -0.05164226551611832;
        bd[33] = 0.013435342414629596;  bd[34] = 0.0;                   bd[35] = 0.0;

        // R4
        bd[36] = -0.0362106806565411;   bd[37] = 0.10540094493378233;   bd[38] = 0.01576433612738956;
        bd[39] = -0.707905442575988;    bd[40] = 0.0;                   bd[41] = 0.7691994139626472;
        bd[42] = -0.1645296432652025;   bd[43] = 0.01828107147391139;   bd[44] = 0.0;

        // R5
        bd[45] = -0.011398193015049775; bd[46] = 0.020437334208704083;  bd[47] = 0.011220896474665617;
        bd[48] = 0.06318369464187532;   bd[49] = -0.6916490244268136;   bd[50] = 0.0;
        bd[51] = 0.7397091390607521;    bd[52] = -0.1479418278121504;   bd[53] = 0.016437980868016712;


        interior = 7;
        d[0] = -0.016666666666666666; d[1] = 0.15; d[2] = -0.75;  
        d[3] = 0;                     d[4] = 0.75; d[5] = -0.15;
        d[6] =  0.016666666666666666;
       
        bhinv[0] = 3.1650670378782326; bhinv[1] = 0.719220844085574;    bhinv[2] = 1.5935079306528956; 
        bhinv[3] = 0.8061205448777757; bhinv[4] = 1.0968642884346833;   bhinv[5] = 0.9862788520810027;

    }

    if ((p!= 2) && (p!= 4) && (p != 6)){
        fprintf(stderr, "%u not a supported order. Code only supports p = 2,4,6\n", p);
        exit(EXIT_FAILURE);
    }
    unsigned int nnz = ((Nxp - 2*bm) * (interior)) + 2*(bm*bn);
    size_t size_D1_v = (nnz)*sizeof(double);
    size_t size_D1_idx = (nnz)*sizeof(unsigned int);

    *D1_nnz = nnz; // Set counts

    size_t size_H_v = Nxp * sizeof(double); // We know this will always be Nxp since diagonal matrix
    size_t size_H_idx = Nxp * sizeof(unsigned int); // We know this will always be Nxp since diagonal matrix
    
    *D1_v = (double *)malloc(size_D1_v);
    *D1_row = (double *)malloc(size_D1_idx);
    *D1_col = (double *)malloc(size_D1_idx);

    *H_v = (double *)malloc(size_H_v);
    *H_row = (double *)malloc(size_H_idx);
    *H_col = (double *)malloc(size_H_idx);


    *HI_v = (double *)malloc(size_H_v);
    *HI_row = (double *)malloc(size_H_idx);
    *HI_col = (double *)malloc(size_H_idx);

    if (*D1_v == NULL || *D1_row == NULL || *D1_col == NULL || *H_v == NULL || *H_row == NULL || *H_col == NULL || *HI_v == NULL || *HI_col == NULL || *HI_row == NULL){
        fprintf(stderr, "Memory Allocation in SBP Operators failed\n");
        exit(EXIT_FAILURE);
    }
  

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

    // Fill H's with appropriate values for SBP operators
    // This is a placeholder. The actual values depend on the specific SBP scheme used.
    for (unsigned int i = 0; i < Nxp; i++){
        (*H_v)[i] = h;
        (*H_row)[i] = i;
        (*H_col)[i] = i;

        if (i < bm){
            (*H_v)[i] = (*H_v)[i] / bhinv[i];
        }

        if (i > Nxp - bm - 1){
            (*H_v)[i] = (*H_v)[i] / bhinv[Nxp - i - 1];
        }
        (*HI_v)[i] = 1.0f / (*H_v)[i];
        (*H_row)[i] = i;
        (*H_col)[i] = i;
    }

    // D1s
    unsigned int idx = 0;

    for (unsigned int i = 0; i < Nxp; i++){

        // Handle Boundary Terms
        if (i < bm){
            for (unsigned int j = 0; j < bn; j++){
                (*D1_v)[idx] = bd[i*bn + j] / h;
                (*D1_row)[idx] = i;
                (*D1_col)[idx] = j;
                idx++;
            }
        }

        else if (i > Nxp - bm - 1){
            for (unsigned int j = 0; j < bn; j++){
                (*D1_v)[idx] = -1.0 * bd[(Nxp - i - 1)*bn + (bn - j - 1)] / h;
                (*D1_row)[idx] = i;
                (*D1_col)[idx] = (Nxp + j - bn);
                idx++;
            }
        }
        else {
            for (unsigned int j=0; j<interior; j++){
                (*D1_v)[idx] =  d[j] / h;
                (*D1_row)[idx] = i;
                (*D1_col)[idx] = i +  j - (interior / 2);
                idx++;
            }
        }
        
    }
}