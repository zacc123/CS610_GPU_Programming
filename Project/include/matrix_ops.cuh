#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#define BLOCK_SIZE 8 // artifact from Coding Assignment 2

/* 
 * Matrix Multiply Kernels. Not Actually used in the CG
 */
__global__ void matrixMulTiledCu(float *a, float *b, float *c, int m, int n, int k);
__global__ void matrixMulCu(float *a, float *b, float *c, int m, int n, int k);

/* 
 * Dense Matrix-Vector Multiply Ab = c
 *  INPUTS: Matrix a, vectors b and c, and dims of A, (m x n)
 */
__global__ void matrixVecCu(double *a, double  *b, double *c, int m, int n);

/* 
 * Dense Matrix-Vector Multiply Ab = c
 *  INPUTS: Matrix a, vectors b and c, and dims of A, (m x n)
 *      OPT: Added tiling with shared memory to share b elements.
 */
__global__ void matrixVecCuOpt(double *a, double  *b, double *c, int m, int n); 

/* 
 * Vector Operations
 */

 /* 
 * Scaling a Vector by a scalar c 
 * IP does the calc in place
 *  INPUTS: Vector a, scalar c, len of vec m
 */
__global__ void VecMulScalarIPCu(double *a, double c, int m);
__global__ void VecMulScalarCu(double *a, double *b, double c, int m);

/* 
 * Adding and subtracting vectors c = a +-b
 * IP does the calc in place
 *  INPUTS: Vectors a,b, and c, len of vec m
 */
__global__ void VecSubCu(double *a, double  *b, double *c, int m);
__global__ void VecAddCu(double *a, double  *b, double *c, int m);

/* 
 * Dot Product a dot b = c
 * First attempt, VecDotCu only does elementwise multiplication: c[i] = a[i]*b[i]
 * SumDotCu, tid 0 sums up the result
 */
__global__ void VecDotCu(double *a, double  *b, double *c, int m);
__global__ void SumDotCu(double *c, int m);

/* 
 * Dot Product a dot b = c
 * Optimized Attempt, Same kernel does element wise multiplication then decreasing stride reduction at a block level.
 */
__global__ void VecDotCuOpt(double *a, double  *b, double *out, int m);

#endif // MATRIX_OPS_CUH