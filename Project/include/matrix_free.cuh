#ifndef MATRIX_FREE_CUH
#define MATRIX_FREE_CUH

/* Struct to hold the SBP-SAT Stencil for CPU Implementations */
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

/* Allocate and Free the D2_mf */
D2_mf *create_D2_mf(int bm, int bn, int interior, int bs, double hi, double *D, double *BS, double *BD);
void freeD2_mf(D2_mf **d2_params);

/* Helper function to apply boundary data to RHS in actual SBP SAT Sim. */
void boundaryMF(double *b, D2_mf *params, double *data, int Nxp);

/* Actual MF action of SBP SAT for CPU
 *      INPUTS: vectors x and result so Ax = result, grid size Nxp, and stencil struct.
 *   
 */
void D2matrixFreeDir(double *x, double *result, int Nxp, D2_mf *d2_params);

/* Actual MF action of SBP SAT for GPU
 * p =2 operator hard coded into the kernel.
 *      INPUTS: vectors x and result so Ax = result, grid size and spacing
 *   
 */
__global__ void D2matrixFreeDirCu(double *x, double *result, int Nxp, double hi);

/* Actual MF action of SBP SAT for GPU
 * p =2 operator hard coded into the kernel.
 * Reduced register usage and number of total operations.
 *      INPUTS: vectors x and result so Ax = result, grid size and spacing
 *   
 */
__global__ void D2matrixFreeDirCuOpt(double *x, double *result, int Nxp, double hi);

#endif // MATRIX_FREE_CUH