#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 2050 /* Exercise 2.3 */
#define THREADS_PER_BLOCK 128

void checkCUDAError(const char*);
void random_ints(int *a);

__global__ void vectorAdd(int *a, int *b, int *c, int max) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	/* Exercise 2.4: Add If statement so only threads in bounds of array do the calculation */
	if (i < max){
		c[i] = a[i] + b[i]; /* Exercise 2.2: Fix Minus Sign */
	} 
	
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void vectorAddCPU(int *a, int *b, int *c, int max) {
	for (unsigned int i=0; i<max; i++){
		c[i] = a[i] + b[i];
	}
	// No return : )
}

int validate(int *cpuResult, int *gpuResult, int max){
	int numErrors = 0;
	for (unsigned int i = 0; i< max; i++){
		if (cpuResult[i] != gpuResult[i]){
			numErrors += 1;
			printf("Error at index %d \nCPU Result: %d | GPU Result: %d\n", i, cpuResult[i], gpuResult[i]);
		}
	}
	
	return numErrors;
}

int main(void) {
	int *a, *b, *c, *c_ref;			// host copies of a, b, c
	int *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size = N * sizeof(int);

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
	int numBlocks = N / THREADS_PER_BLOCK;
	if (N % THREADS_PER_BLOCK != 0){
		numBlocks++;
	}

	vectorAdd <<<numBlocks, THREADS_PER_BLOCK >>>(d_a, d_b, d_c, N);
	
	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	checkCUDAError("CUDA kernel");

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	/* Calculate CPU array and validate */
	vectorAddCPU(a, b, c_ref, N);
	errors = validate(c_ref, c, N);
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
	for (unsigned int i = 0; i < N; i++){
		a[i] = rand();
	}
}


