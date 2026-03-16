#ifndef TESTS_CUH
#define TESTS_CUH

/* Basic Correctness Tests for CG */
void run_cg_test(double *vec, int k);
void run_cg_test_cu(double *vec, int k);

/* Helper for outputting results */
void write_time_to_file(double timer[], char *filename, char* operation, bool Opt, int k, bool write, int threads);

/* Run Tests on each set of kernels
 * Each runs experiments 5 times, using CPU for validation
 * Correctness results are output in Stdout, and timing saved to file.
 */
void test_BLAS_ops(int p, int k, int threadsPerBlock, bool write);
void test_MF_ops(int k, int threadsPerBlock, bool write);
void test_cg_dense(int k, int threadsPerBlock, bool write);
void test_cg_mf(int k, int threadsPerBlock, bool write);


int elementwiseValidate(double *a, double *b, int k, double tol);

#include <curand_kernel.h>

// Kernel to initialize the random number generator states
__global__ void setup_kernel(curandState *state, unsigned long long seed);

// Kernel to generate random numbers and store them in an array
__global__ void generate_kernel(curandState *state, double *rand_nums, int n);

void generate_random_array(int k, double *A);
#endif // TESTS_CUH