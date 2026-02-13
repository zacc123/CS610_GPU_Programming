#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

#define M 1000// C Matrix Row Dim
#define N 1200// C Matrix Col Dim
#define K 500 // Other dim for A and B Matrix (m, k) x (k x n)
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16
#define R_MAX = 10000 // Use this to make random float
#define EPS 1e-5 // Use this for matching tolerance
#define BLOCK_SIZE 16

#define NUM_TIMERS 2
#define GPU_TIME 0
#define CPU_TIME 1


static inline uint64_t ReadTSC(void)
{
#if defined(__i386__)

    uint64_t x;
    __asm__ __volatile__(".byte 0x0f, 0x31":"=A"(x));
    return x;

#elif defined(__x86_64__)

    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc":"=a"(lo), "=d"(hi));
    return ((uint64_t) lo) | (((uint64_t) hi) << 32);

#elif defined(__powerpc__)

    uint64_t result = 0;
    uint64_t upper, lower, tmp;
    __asm__ __volatile__("0:                  \n"
                         "\tmftbu   %0           \n"
                         "\tmftb    %1           \n"
                         "\tmftbu   %2           \n"
                         "\tcmpw    %2,%0        \n"
                         "\tbne     0b         \n":"=r"(upper), "=r"(lower),
                         "=r"(tmp)
        );
    result = upper;
    result = result << 32;
    result = result | lower;
    return result;

#endif // defined(__i386__)
}

void checkCUDAError(const char*);
void random_floats(float *a, int m, int n);
void print_time(double timer[]);
void InitTSC(void);
double ElapsedTime(uint64_t ticks);
int validate(float *cpuResult, float *gpuResult, int m, int n);
void matrixMulCPU(float *a, float *b, float *c, int m, int n, int k);

__global__ void matrixMul(float *a, float *b, float *c, int m, int n, int k) {
	
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
	
	if (idx_x < m && idx_y < n){
		c[idx_c] = 0.0f; // reset ahead of time
	}

    /* Now Start Load and Calc Loop */
	for (unsigned int i = 0; i < num_loads; i++){

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
			for (unsigned int j = 0; j < BLOCK_SIZE; j++){
				c[idx_c] += a_shared[j + threadIdx.x*BLOCK_SIZE] * b_shared[j*BLOCK_SIZE + threadIdx.y];
			}
		}
		__syncthreads(); // Make sure everything copies over
				
	}
		
	
}


int main(void) {
	float *a, *b, *c, *c_ref;			// host copies of a, b, c
	float *d_a, *d_b, *d_c;			// device copies of a, b, c
	int errors;
	unsigned int size_a = M * K * sizeof(float); // Now handles 2D matrix size NxM
	unsigned int size_b = K * N * sizeof(float); 
	unsigned int size_c = M * N * sizeof(float); 

	
	/* Adding Timing Code from my 531 Class */
	// Initialize timess
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();

	// Alloc space for device copies of a, b, c
	cudaMalloc((void **)&d_a, size_a);
	cudaMalloc((void **)&d_b, size_b);
	cudaMalloc((void **)&d_c, size_c);
	checkCUDAError("CUDA malloc");

	// Alloc space for host copies of a, b, c and setup input values
	a = (float *)malloc(size_a); random_floats(a, M, K);
	b = (float *)malloc(size_b); random_floats(b, K, N);
	c = (float *)malloc(size_c); 
	c_ref = (float *)malloc(size_c); // For CPU Validation

	// Copy inputs to device
	cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
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
	dim3 numThreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

	t0 = ReadTSC();
	matrixMul <<<numBlocks, numThreads >>>(d_a, d_b, d_c, M, N, K);

	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	timer[GPU_TIME] += ElapsedTime(ReadTSC() - t0); // Dont include error check in time
	checkCUDAError("CUDA kernel");
	

	// Copy result back to host
	cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy");

	/* Calculate CPU array and validate */
	t0 = ReadTSC();
	matrixMulCPU(a, b, c_ref, M, N, K);
	timer[CPU_TIME] += ElapsedTime(ReadTSC() - t0);
	errors = validate(c_ref, c, M, N);
	printf("Ran Validation and found ... %d error out of %d entries", errors, M*N);

	// Cleanup
	free(a); free(b); free(c); free(c_ref);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	checkCUDAError("CUDA cleanup");

	print_time(timer);

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

void random_floats(float *a, int m, int n)
{
    /* Generates Random Float between 0 and 1 
		For mxn matrix*/
	
	for (unsigned int i = 0; i < m*n; i++){
		a[i] = (float)rand() / (float)RAND_MAX;;
	}
}

/* Exercise 1.7: CPU Version of Matrix Mul and validate */
void matrixMulCPU(float *a, float *b, float *c, int m, int n, int k) {
	int idx;
	unsigned int a_idx, b_idx;

	for (unsigned int i=0; i<m; i++){
		for (unsigned int j=0; j<n; j++){
			
			idx = j + (i)*n; // Calc Idx of Array
			c[idx] = 0;		 // Reset to 0
			for (unsigned int _k=0; _k<k; _k++){
				a_idx = (i * k) + _k;
				b_idx = (_k * n) + j;
				c[idx] += a[a_idx] * b[b_idx];
			}
		}
	}
	// No return : )
}

int validate(float *cpuResult, float *gpuResult, int m, int n){
	int numErrors = 0;
    int idx; // 1D idx to be calculated

	double tolerance = EPS * (double)K;
	

	for (unsigned int i = 0; i< m; i++){
        for (unsigned int j=0; j<n; j++){
            idx = (i * n) + j;
            if (abs(cpuResult[idx] - gpuResult[idx]) > tolerance) {
                numErrors++;
                printf("Error at index [%d %d] \nCPU Result: %f | GPU Result: %f\n", i, j, cpuResult[idx], gpuResult[idx]);
            }
        }
	}
	printf("Tolerance: %f\n", tolerance);
	return numErrors;
}


/* Print timing information 
 */
void print_time(double timer[])
{
    fprintf(stdout, "\nDevice\t\tTime (Sec)\n");
    fprintf(stdout, "GPU\t\t");
    fprintf(stdout, "%f\n", timer[GPU_TIME]);
	long Glops = (long) M * K * N * 2 / 1e9;
	double Gflops =  (double) Glops / timer[GPU_TIME];
	fprintf(stdout, "GFLOPS per Sec:\t %f\n", Gflops);
    fprintf(stdout, "CPU\t\t");
    fprintf(stdout, "%f\n", timer[CPU_TIME]);

}



static double g_ticks_persecond = 0.0;

void InitTSC(void)
{
    uint64_t start_tick = ReadTSC();
    sleep(1);
    uint64_t end_tick = ReadTSC();

    g_ticks_persecond = (double) (end_tick - start_tick);
    //fprintf(stderr, "%e ticks per second.\n", g_ticks_persecond);
}


double ElapsedTime(uint64_t ticks)
{
    if (g_ticks_persecond == 0.0) {
        fprintf(stderr, "TSC timer has not been initialized.\n");
        return 0.0;
    }
    else {
        return (ticks / g_ticks_persecond);
    }
}