#ifndef CG_CUH
#define CG_CUH

/*
 * File where CG implementations live
 */ 
#include "../include/matrix_free.cuh"

/*
 * CPU Implementatons to solve Ax = b for x.
 */ 

/* 
 * 1st CG Attempt, with inefficient operation ordering and memory usage 
 *  Inputs: Matrix A, vectors x, b, len of vectors k, and matching tolerance and max iterations.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradient(double *A, double *x, double *b, int k, double atol, int max_iter);

/* 
 * 2nd CG Attempt for Dense ops on the CPU. Better memory usage and convergence;
 * This is the CPU Baseline 
 *  Inputs: Matrix A, vectors x, b, len of vectors k, and matching tolerance and max iterations.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradient2(double *A, double *x, double *b, int k, double atol, int max_iter);

/* 
 * Matrix Free CPU Version. Identical to CG2, but with a matrix free Ax calc. Need to pass in 
 *     Inputs:  vectors x, b, len of vectors k, and matching tolerance and max iterations, and add D2_params that holds A stencil.
 *     Outputs: Num Iters used to converge.
 */
int conjugateGradientMFCpu(double *x, double *b, int k, double atol, int max_iter, D2_mf *d2_params);

/* - - - - - - - - - - - - - - 
 * CUDA Implementations
 - - - - - - - - - - - - - - -*/

/* 
 * GPU Enabled CG. Handles Memory copy, allocation, and then solves CG on the GPU. 
 * Uses naive kernels as baseline.
 *  Inputs: Matrix A, vectors x, b, len of vectors k, and matching tolerance and max iterations, and threadsPerBlock.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradientCu(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock);

/* 
 * GPU Enabled CG. Handles Memory copy, allocation, and then solves CG on the GPU. 
 * Uses Optimized kernels.
 *  Inputs: Matrix A, vectors x, b, len of vectors k, and matching tolerance and max iterations, and threadsPerBlock.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradientCuOpt(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock);

/* 
 * GPU Enabled CG. Follows form of above, but use a MF stencil for A. 
 * Uses naive kernels + a MF kernel.
 *  Inputs: vectors x, b, grid_spacing hi, len of vectors k, and matching tolerance and max iterations, and threadsPerBlock.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradientCuMF(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock);


/* 
 * GPU Enabled CG. Follows form of above, but use a MF stencil for A. 
 * Uses Optimized kernels + a optimized MF kernel.
 *  Inputs: vectors x, b, grid_spacing hi, len of vectors k, and matching tolerance and max iterations, and threadsPerBlock.
 *  Outputs: Num Iters used to converge.
 */
int conjugateGradientCuMFOpt(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock);

#endif // MATRIX_OPS_CUH