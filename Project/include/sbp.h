#ifndef SBP_H
#define SBP_H
#pragma once

#include "utils.h"   // makes SparseMatrixCOO visible

/*
 * Function to build and destroy Dense 1st and 2nd Derivative Operators and Necessary Matrices
 * IMPORTANT NOTES: p is the order p in {2, 4, 6}
 */
void create_1D_D1_operator_dense(double **D1, double **H, double **HI, int Nx, int p, float X0, float XN);
void free_D1_ops(double **D1, double **H, double **HI);

void create_1D_D2_operator_dense(double **D2, double **BS, int Nx, int p, float X0, float XN);
void free_D2_ops(double **D2, double **BS);

/*
 * SAT term creation for 2 DIR Conditions using outputs from above operators.
 */
void addSAT_dir(double ** SAT, double *BS, int Nx, double *HI, double mu);


/* 
 * Helpers to handle boundary data and source term 
 */
void data(double *g, float *domain, double t, int Nxp);
void source(double *u, float *domain, double t, int Nxp);
void boundary(double *b, double *SAT, double *data, int Nxp);

/* 
 * Helper to build grid spacing matrix compatible with MF approach, to use in the above. 
 */
void create_H_Mf(double **H, double **HI, int Nx, int p, float X0, float XN);
#endif // SBP_H