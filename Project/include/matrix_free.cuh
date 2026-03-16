#ifndef MATRIX_FREE_CUH
#define MATRIX_FREE_CUH

#define BLOCK_SIZE 8
#define BATCH_SIZE 10 // This will be num of matrice's
#define NUM_STREAM 3

typedef struct {
    int bm;
    int bn;
    int interior;
    int bs;
    double hi;
    double *D;
    double *BS;
    double *BD;
} D2_mf;

D2_mf *create_D2_mf(int bm, int bn, int interior, int bs, double hi, double *D, double *BS, double *BD);
void freeD2_mf(D2_mf **d2_params);
void boundaryMF(double *b, D2_mf *params, double *data, int Nxp);
void D2matrixFreeDir(double *x, double *result, int Nxp, D2_mf *d2_params);

__global__ void D2matrixFreeDirCuOpt(double *x, double *result, int Nxp, double hi);

__global__ void D2matrixFreeDirCu(double *x, double *result, int Nxp, double hi);
#endif // MATRIX_FREE_CUH