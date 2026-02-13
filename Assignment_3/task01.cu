#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

#define M 10000// C Matrix Row Dim
#define N 10000// C Matrix Col Dim
#define K 10000 // Other dim for A and B Matrix (m, k) x (k x n)
#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define R_MAX = 10000 // Use this to make random float
#define EPS 1e-5 // Use this for matching tolerance
#define BLOCK_SIZE 8
#define BATCH_SIZE 10 // This will be num of matrice's
#define PIPELINE 0
#define VALIDATE 0
#define NUM_STREAM 3


/****************************************************/
/* Timing Infrastructure from Prof Choi's 531 Class */
/****************************************************/

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
/************************************************************************/
/************************************************************************/
/************************************************************************/

/************************************************************************/
/* Start 610 Code                                                       */
/************************************************************************/

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
	
    float thread_total = 0.0;

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
                thread_total += a_shared[j + threadIdx.x*BLOCK_SIZE] * b_shared[j*BLOCK_SIZE + threadIdx.y];
			}
		}
		__syncthreads(); // Make sure everything copies over
				
	}

    if (idx_x < m && idx_y < n){
        c[idx_c] = thread_total;
    }
		
	
}

int main(void) {

    /* Main structural Difference from HW 2 is a, b, c will be an array of pointers,
        pointing to matrix array */
	
	int errors;
	unsigned int size_a = M * K * sizeof(float); // Now handles 2D matrix size NxM
	unsigned int size_b = K * N * sizeof(float); 
	unsigned int size_c = M * N * sizeof(float); 

    // Array Setup on Host
	float *A[BATCH_SIZE];
    float *B[BATCH_SIZE];
    float *C[BATCH_SIZE];
    float *C_ref[BATCH_SIZE];
    
    /*
    for (unsigned int i = 0; i<BATCH_SIZE; i++){
        
        A[i] = (float *)malloc(size_a); random_floats(A[i], M, K);
        B[i] = (float *)malloc(size_b); random_floats(B[i], K, N);
        C[i] = (float *)malloc(size_c); 
        C_ref[i] = (float *)malloc(size_c); 
    }
    */
    for (unsigned int i = 0; i < BATCH_SIZE; i++){
        cudaHostAlloc((void **)&A[i], size_a, cudaHostAllocDefault); random_floats(A[i], M, K);
        cudaHostAlloc((void **)&B[i], size_b, cudaHostAllocDefault); random_floats(B[i], K, N);
        cudaHostAlloc((void **)&C[i], size_c, cudaHostAllocDefault);
        C_ref[i] = (float *)malloc(size_c); 
    }


	/* Adding Timing Code from my 531 Class */
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();

    // Array Setup on Device
	// Allocate only what we need on device at a given time (2 Max)
    // 1 can be copied into while the other computes and copies out
	float *d_a0, *d_b0, *d_c0, *d_a1, *d_b1, *d_c1, *d_a2, *d_b2, *d_c2;
    cudaMalloc((void **)&d_a0, size_a);
	cudaMalloc((void **)&d_b0, size_b);
	cudaMalloc((void **)&d_c0, size_c);
    cudaMalloc((void **)&d_a1, size_a);
	cudaMalloc((void **)&d_b1, size_b);
	cudaMalloc((void **)&d_c1, size_c);
    cudaMalloc((void **)&d_a2, size_a);
	cudaMalloc((void **)&d_b2, size_b);
	cudaMalloc((void **)&d_c2, size_c);
	checkCUDAError("CUDA malloc");

    float *d_a[3] = {d_a0, d_a1, d_a2};
    float *d_b[3] = {d_b0, d_b1, d_b2};
    float *d_c[3] = {d_c0, d_c1, d_c2};

	// Setup Kernel Launch Params
	int numBlocksX = (M + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
	int numBlocksY = (N + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;

	dim3 numBlocks(numBlocksX, numBlocksY, 1);
	dim3 numThreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);

    // setup streams
    cudaStream_t stream[NUM_STREAM]; // store in an array to make next part easier>
    for (unsigned int j = 0; j<NUM_STREAM; j++){
        cudaStreamCreate(&stream[j]);
    }

    cudaDeviceSynchronize();
    printf("First Mem Copy:\n");
    // Now the fun starts with copying and computing

    // Build in a warm up
    cudaMemcpyAsync(d_a0, A[0], size_a, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_b0, B[0], size_b, cudaMemcpyHostToDevice, stream[0]);
    matrixMul <<<numBlocks, numThreads, 0, stream[0]>>>(d_a[0], d_b[0], d_c[0], M, N, K);
    cudaDeviceSynchronize();


	t0 = ReadTSC(); // Time the whole Batched Process
    // Copy first array over
    cudaMemcpyAsync(d_a0, A[0], size_a, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_b0, B[0], size_b, cudaMemcpyHostToDevice, stream[0]);

    // Now loop through:
    int idx;
    if (PIPELINE){
        printf("Using STREAMS:\n");
        for (unsigned int k=0; k<BATCH_SIZE; k++){
            
             idx = k%NUM_STREAM;
            // Get next copy over if appropriate
                if (k != 0){
                        cudaMemcpyAsync(d_a[idx], A[k], size_a, cudaMemcpyHostToDevice, stream[idx]);
                        cudaMemcpyAsync(d_b[idx], B[k], size_b, cudaMemcpyHostToDevice, stream[idx]);
                }

                matrixMul <<<numBlocks, numThreads, 0, stream[idx]>>>(d_a[idx], d_b[idx], d_c[idx], M, N, K);
                    //printf("D2H Array 0\n");
                cudaMemcpyAsync(C[k], d_c[idx], size_c, cudaMemcpyDeviceToHost, stream[idx]);
        }
    }
    // No Batching, just need the 1 set of arrays
    else {
        for (unsigned int k=0; k<BATCH_SIZE; k++){
            
            if (k != 0){
                cudaMemcpy(d_a0, A[k], size_a, cudaMemcpyHostToDevice);
                cudaMemcpy(d_b0, B[k], size_b, cudaMemcpyHostToDevice);
            }

            matrixMul <<<numBlocks, numThreads, 0, stream[0]>>>(d_a0, d_b0, d_c0, M, N, K);
            //printf("D2H Array 0\n");
            cudaMemcpy(C[k], d_c0, size_c, cudaMemcpyDeviceToHost);
    
        }
    }
	
	/* wait for all threads to complete */
	cudaDeviceSynchronize();
	timer[GPU_TIME] += ElapsedTime(ReadTSC() - t0); // Dont include error check in time
	checkCUDAError("CUDA kernel");
	
	/* Calculate CPU array and validate */
	t0 = ReadTSC();
	timer[CPU_TIME] += ElapsedTime(ReadTSC() - t0);

    // Verification
    int local_error = 0;
    errors = 0;
    // Getting tired of manually toggling
    if (VALIDATE){
        for (unsigned int i=0; i<BATCH_SIZE; i++){
            matrixMulCPU(A[i], B[i], C_ref[i], M, N, K);
            local_error = validate(C_ref[i], C[i], M, N);
            printf("Errors in Mul %d = %d\n", i, local_error);
            errors += local_error;
        }
        printf("For Batch Size: %d\n", BATCH_SIZE);
        printf("Ran Validation and found ... %d error out of %d entries", errors,BATCH_SIZE*M*N);
    }

	// Cleanup
    for (unsigned int i = 0; i<BATCH_SIZE; i++){
        
        //free(A[i]); A[i] = NULL;
        //free(B[i]); B[i] = NULL;
        //free(C[i]); C[i] = NULL;
        cudaFreeHost(A[i]);
        cudaFreeHost(B[i]);
        cudaFreeHost(C[i]);
        free(C_ref[i]); C_ref[i] = NULL;
    }

    // Device Mem Allocations
    cudaFree(d_a0); cudaFree(d_b0); cudaFree(d_c0);
	cudaFree(d_a1); cudaFree(d_b1); cudaFree(d_c1); 
    cudaFree(d_a2); cudaFree(d_b2); cudaFree(d_c2);

    // Destroy streams
    for (unsigned int j = 0; j<NUM_STREAM; j++){
        cudaStreamDestroy(stream[j]);
    }
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
                // printf("Error at index [%d %d] \nCPU Result: %f | GPU Result: %f\n", i, j, cpuResult[idx], gpuResult[idx]);
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
	double Gflops =  (double) Glops * BATCH_SIZE / timer[GPU_TIME];
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