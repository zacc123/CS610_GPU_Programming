#ifndef SBP_H
#define SBP_H

void create_1D_D1_operator_dense(double **D1, double **H, double **HI, unsigned int Nx, unsigned int p, float X0, float XN);
void free_D1_ops(double **D1, double **H, double **HI);

void create_1D_D2_operator_dense(double **D2, double **BS, unsigned int Nx, unsigned int p, float X0, float XN);
void free_D2_ops(double **D2, double **BS);

void addSAT_dir(double ** SAT, double *BS, unsigned int Nx, double *HI, double mu);

void data(double *g, float *domain, double t, unsigned int Nxp);
void source(double *u, float *domain, double t, unsigned int Nxp);

void boundary(double *b, double *SAT, double *data, unsigned int Nxp);


// Sparse operators
void create_1D_D1_operator_sparse(double **D1_v, double **D1_row, double **D1_col, unsigned int *D1_nnz, double **H_v, double **H_row, double **H_col, double **HI_v, double **HI_row, double **HI_col, unsigned int Nx, unsigned int p, float X0, float XN);

#endif // SBP_H