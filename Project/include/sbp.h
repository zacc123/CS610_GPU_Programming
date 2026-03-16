#ifndef SBP_H
#define SBP_H
#pragma once

#include "utils.h"   // makes SparseMatrixCOO visible

void create_1D_D1_operator_dense(double **D1, double **H, double **HI, int Nx, int p, float X0, float XN);
void free_D1_ops(double **D1, double **H, double **HI);

void create_1D_D2_operator_dense(double **D2, double **BS, int Nx, int p, float X0, float XN);
void free_D2_ops(double **D2, double **BS);

void addSAT_dir(double ** SAT, double *BS, int Nx, double *HI, double mu);

void data(double *g, float *domain, double t, int Nxp);
void source(double *u, float *domain, double t, int Nxp);

void boundary(double *b, double *SAT, double *data, int Nxp);


// Sparse operators
void create_1D_D1_operator_sparse(SparseMatrixCOO **D1, SparseMatrixCOO **H, SparseMatrixCOO **HI,
    int Nx, int p, float X0, float XN);
  
void create_1D_D2_operator_sparse(SparseMatrixCOO **D2, SparseMatrixCOO **BS, 
                                int Nx, int p, float X0, float XN);

void create_H_Mf(double **H, double **HI, int Nx, int p, float X0, float XN);
#endif // SBP_H