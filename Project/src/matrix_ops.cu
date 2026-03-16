#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/matrix_ops.cuh"
// #include <cuda_runtime.h>

__global__ void matrixMulTiledCu(float *a, float *b, float *c, int m, int n, int k) {
	
    // Start by Setting up shared mem

    __shared__ float a_shared[BLOCK_SIZE*BLOCK_SIZE]; // Make up for 2D Array
    __shared__ float b_shared[BLOCK_SIZE*BLOCK_SIZE]; //

	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x); // Global tids
	int idx_y = threadIdx.y + (blockDim.y * blockIdx.y); 
    
	/* Only Do Operations for TIds in Bounds */
	
		
	int idx_s = BLOCK_SIZE * threadIdx.x + threadIdx.y; // Tid for shared memory in a block
    int idx_c = idx_x * n + idx_y; // Tid for C array
    int num_loads = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;

	int a_idx, b_idx, a_col, b_row;
	
    float thread_total = 0.0;

	if (idx_x < m && idx_y < n){
		c[idx_c] = 0.0f; // reset ahead of time
	}

    /* Now Start Load and Calc Loop */
	for (int i = 0; i < num_loads; i++){

		a_idx = (idx_x * k) + (threadIdx.y) + (i*BLOCK_SIZE); // I think ??? 
		b_idx = idx_y + (threadIdx.x + i*BLOCK_SIZE)*n;

		a_col = (threadIdx.y) + (i*BLOCK_SIZE); // Guard rail for going right
		b_row = (threadIdx.x + i*BLOCK_SIZE); // Guard rail for going down
			
		if (a_col < k && idx_x < m){ // Now some blocks might have some hang over
			a_shared[idx_s] = a[a_idx];
		}
		else {
			a_shared[idx_s] = 0.0f;
		}
		if (b_row < k && idx_y < n){ // Now some blocks might have some hang over
			b_shared[idx_s] = b[b_idx];
		}
		else {
			b_shared[idx_s] = 0.0f;
		}

		__syncthreads(); // Make sure everything copies over
		if (idx_x < m && idx_y < n){
			/* Do the dot product through the block */
			for (int j = 0; j < BLOCK_SIZE; j++){
                thread_total += a_shared[j + threadIdx.x*BLOCK_SIZE] * b_shared[j*BLOCK_SIZE + threadIdx.y];
			}
		}
		__syncthreads(); // Make sure everything copies over
				
	}

    if (idx_x < m && idx_y < n){
        c[idx_c] = thread_total;
    }
}

/*
 *
 *
 * Naive GPU Implementations
 *
 *
 *
 */

// Naive Mem Kernel
// Assume 1 thread per element
__global__ void matrixMulCu(double *a, double *b, double *c, int m, int n, int k) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	int idx_y = threadIdx.y + (blockDim.y * blockIdx.y);
	/* Exercise 2.4: Add If statement so only threads in bounds of array do the calculation */
	if (idx_x < m && idx_y < n){
		
		int c_idx = idx_x * n + idx_y;
		int a_idx, b_idx;
		c[c_idx] = 0.0f; // reset ahead of time
		
		for (int i = 0; i < k; i++){
			a_idx = (idx_x * k) + i; 
			b_idx = idx_y + (i * n);
			c[c_idx] += a[a_idx] * b[b_idx];
		}
		
	}
}

// Naive Matrix Vec
// Assume 1 thread per element in output vector
// Ab = c
__global__ void matrixVecCu(double *a, double  *b, double *c, int m, int n) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	
	if (idx_x < m){
		double result = 0.0;
        for (int i = 0; i<n; i++){
            result += a[i + idx_x*n] * b[i];
        }
        c[idx_x] = result;
	}
}

__global__ void VecSubCu(double *a, double  *b, double *c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx_x < m){
        c[idx_x] = a[idx_x] - b[idx_x];
	}
}

__global__ void VecAddCu(double *a, double  *b, double *c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx_x < m){
        c[idx_x] = a[idx_x] + b[idx_x];
	}
}

__global__ void VecDotCu(double *a, double  *b, double *c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx_x < m){
        c[idx_x] = a[idx_x]*b[idx_x];
	}
}

__global__ void SumDotCu(double *c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	double sum = 0.0;
	if (idx_x < 1){
		
        for (int i = 0; i < m; i++){
			sum += c[i];
		}
	}
	__syncthreads();
	
	if (idx_x < 1){
       c[0] = sum;
	}
	
}

__global__ void VecMulScalarIPCu(double *a, double c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx_x < m){
        a[idx_x] = c *  a[idx_x];
	}
}

__global__ void VecMulScalarCu(double *a, double *b, double c, int m) {
	
	int idx_x = threadIdx.x + (blockDim.x * blockIdx.x);
	if (idx_x < m){
        b[idx_x] = c *  a[idx_x];
	}
}

/*
 *
 *
 * Optimized Versions
 *
 *
 */
 // Opt Strategy, give each block shared 2 shared mem arrays
	// 1 for Block Level Reduction (decreasing stride)
	
__global__ void VecDotCuOpt(double *a, double  *b, double *c, int m) {
	
	extern __shared__ double block_mem[]; // should be size of block ...
	
	int glob_idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int local_idx = threadIdx.x;
	
	if (glob_idx < m){
        block_mem[local_idx] = a[glob_idx]*b[glob_idx];
	}
	else {
		block_mem[local_idx] = 0.0;
	}

	// Reduce each block
	for (int stride = blockDim.x/2; stride >= 1; stride /= 2){
		if (local_idx < stride){
			block_mem[local_idx] += block_mem[local_idx + stride];
		}
		__syncthreads();
	}

	if (local_idx == 0){
		atomicAdd(c, block_mem[0]);
	}
}

// More Efficient
// Assume 1 thread per element in output vector
// Ab = c
__global__ void matrixVecCuOpt(double *a, double  *b, double *c, int m, int n) {
	
	// Assume Useful Shared Mem size is whole block ceil(blocksize/8) transactions per
	extern __shared__ double block_mem[];

	int glob_idx = threadIdx.x + (blockDim.x * blockIdx.x);
	int local_idx = threadIdx.x;
	
	int stride = blockDim.x;
	double result = 0.0;

	int tile_width = 0;
	for (int offset = 0; offset < m; offset += stride){
		// Load in shared part
		if (offset + local_idx < m){
			block_mem[local_idx] = b[local_idx + offset];
		}
		else {
			block_mem[local_idx] = 0.0;
		}
		__syncthreads();

		if (glob_idx < m){
			tile_width = min(stride, m - offset);
			for (int i = 0; i<tile_width; i++){
				result += a[i +offset + glob_idx*n] * block_mem[i];
			}
		}
		__syncthreads();
	}
	if (glob_idx < m){
		c[glob_idx] = result;
	}
}