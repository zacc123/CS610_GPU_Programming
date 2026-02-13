#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M 3048 /* Exercise 3.1 */
#define N 2005 /* Exercise 3.1 */
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void matrixAdd(int *a, int *b, int *c, int max) {
	
	
	int idx = threadIdx.x + (blockDim.x * threadIdx.y) + (blockIdx.x * (blockDim.y * blockDim.x)) + (blockIdx.y * (gridDim.x * blockDim.y * blockDim.x));

	/* Exercise 2.4: Add If statement so only threads in bounds of array do the calculation */
	if (idx < max){
		c[idx] = a[idx] + b[idx]; /* Exercise 2.2: Fix Minus Sign */
	} 
	
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixAddCPU(int *a, int *b, int *c, int m, int n) {
	int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			c[idx] = a[idx] + b[idx];
		}
		
	}
	// No return : )
}

int validate(int *cpuResult, int *gpuResult, int n, int m){
	int numErrors = 0;
    int idx; // 1D idx to be calculated

	for (unsigned int i = 0; i< n; i++){
        for (unsigned int j=0; j<m; j++){
            idx = (i * m) + j;
            if (cpuResult[idx] != gpuResult[idx]) {
                numErrors++;
                printf("Error at index [%d %d] \nCPU Result: %d | GPU Result: %d\n", i, j, cpuResult[idx], gpuResult[idx]);
            }
        }
	}
	return numErrors;
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * M * sizeof(int); // Now handles 2D matrix size NxM

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (int *)malloc(size); random_ints(a);
	b = (int *)malloc(size); random_ints(b);
	c = (int *)malloc(size);
	c_ref = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");

	// Launch add() kernel on GPU
	int numBlocksX = (M) / THREADS_PER_BLOCK_X;
	if ((M) % THREADS_PER_BLOCK_X != 0){
		numBlocksX++;
	}

	// Launch add() kernel on GPU
	int numBlocksY = (N) / THREADS_PER_BLOCK_Y;
	if ((N) % THREADS_PER_BLOCK_Y != 0){
		numBlocksY++;
	}

	dim3 numBlocks(numBlocksX, numBlocksY, 1);

	matrixAdd <<<numBlocks, THREADS_PER_BLOCK_X*THREADS_PER_BLOCK_Y >>>(d_a, d_b, d_c, N*M);
	
	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	/* Calculate CPU array and validate */
	matrixAddCPU(a, b, c_ref, M, N);
	errors = validate(c_ref, c, M, N);
	printf("Ran Validation and found ... %d errors", errors);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void random_ints(int *a)
{
    // Now does N^2 matrix
	for (unsigned int i = 0; i < N*M; i++){
		a[i] = rand();
	}
}


