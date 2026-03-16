#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#define BLOCK_SIZE 8
#define BATCH_SIZE 10 // This will be num of matrice's
#define NUM_STREAM 3


// Matrix Ops
__global__ void matrixMulTiledCu(float *a, float *b, float *c, int m, int n, int k);
__global__ void matrixMulCu(float *a, float *b, float *c, int m, int n, int k);

// Matrix-Vec Ops
__global__ void matrixVecCu(double *a, double  *b, double *c, int m, int n);

// Vec Ops
__global__ void VecMulScalarIPCu(double *a, double c, int m);
__global__ void VecMulScalarCu(double *a, double *b, double c, int m);
__global__ void VecSubCu(double *a, double  *b, double *c, int m);
__global__ void VecAddCu(double *a, double  *b, double *c, int m);
__global__ void VecDotCu(double *a, double  *b, double *c, int m);
__global__ void SumDotCu(double *c, int m);
__global__ void VecDotCuOpt(double *a, double  *b, double *out, int m);
__global__ void matrixVecCuOpt(double *a, double  *b, double *c, int m, int n); 
#endif // MATRIX_OPS_CUH