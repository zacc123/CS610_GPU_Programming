
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/tests.cuh"
#include "../include/utils.h"
#include "../include/cg.cuh"
#include "../include/sbp.h"
#include "../include/matrix_ops.cuh"
#include "../include/timing.h"

// Timing Infrastructure from Prof. Choi's class
#define NUM_TIMERS 10
#define GPU_TIME_0 0
#define GPU_TIME_1 1
#define GPU_TIME_2 2
#define GPU_TIME_3 3
#define GPU_TIME_4 4
#define CPU_TIME_1 5
#define CPU_TIME_2 6
#define CPU_TIME_3 7
#define CPU_TIME_4 8
#define CPU_TIME_5 9

#define EPS 1e-6;

void run_cg_test(double *vec, int k){
    // 2 Tests, 1 Ix = vec
    // A (rand ish) * randish vec
    // set up Id matrix
    double *A = (double *)malloc(k*k*sizeof(double));
    double *AT = (double *)malloc(k*k*sizeof(double));
    double *ASpd = (double *)malloc(k*k*sizeof(double));
    
    for (int i = 0; i < k*k; i++){
        A[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }

    matrixTranspose(A, AT, k, k);
    matrixMulDense(AT, A, ASpd, k, k, k); // Now A is spd
    
    
    double *vec_A = (double *)malloc(k*sizeof(double));
    for (int i = 0; i < k; i++){
        vec_A[i] = (double)rand() / (double)RAND_MAX;
    }

    double *res_A = (double *)malloc(k*sizeof(double));
    matrixVecDense(ASpd, vec_A, res_A, k, k);

    double *x_A = (double *)calloc(k, sizeof(double)); // x0 = 0

    double *I = (double *)malloc(k*k*sizeof(double));
    memset(I, 0, sizeof(double) * k*k);
    for (int i = 0; i < k; i++){
        I[i + k*i] = 1.0;
    }
    
    double *x = (double *)malloc(k*sizeof(double));
    memset(x, 0, sizeof(double) * k);

    double atol = 1e-6;
    int max_iter = 1500;
    int iters = conjugateGradient(I, x, vec, k, atol, max_iter);
    int iters_A = conjugateGradient(ASpd, x_A, res_A, k, atol, max_iter);

    double error = 0.0;
    for (int i = 0; i < k; i++){
        error += sqrt((vec[i] - x[i])*(vec[i] - x[i]));
    }

    double errorA = 0.0;
    for (int i = 0; i < k; i++){
        errorA += sqrt((vec_A[i] - x_A[i]) * (vec_A[i] - x_A[i]));
    }
    printf("I Test: CG Finished with error: %f after %d iterations \n", error, iters);
    printf("A Test: CG Finished with error: %.12f after %d iterations \n", errorA, iters_A);


    free(I); I = NULL;
    free(x); x=NULL;
    free(A); free(AT);
    free(ASpd); free(vec_A);
    free(x_A); free(res_A);

}

void run_cg_test_cu(double *vec, int k){
    // 2 Tests, 1 Ix = vec
    // A (rand ish) * randish vec
    // set up Id matrix
    double *A = (double *)malloc(k*k*sizeof(double));
    double *AT = (double *)malloc(k*k*sizeof(double));
    double *ASpd = (double *)malloc(k*k*sizeof(double));
    
    for (int i = 0; i < k*k; i++){
        A[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }

    matrixTranspose(A, AT, k, k);
    matrixMulDense(AT, A, ASpd, k, k, k); // Now A is spd
    
    
    double *vec_A = (double *)malloc(k*sizeof(double));
    for (int i = 0; i < k; i++){
        vec_A[i] = (double)rand() / (double)RAND_MAX;
    }

    double *res_A = (double *)malloc(k*sizeof(double));
    matrixVecDense(ASpd, vec_A, res_A, k, k);

    double *x_A = (double *)calloc(k, sizeof(double)); // x0 = 0

    double *I = (double *)malloc(k*k*sizeof(double));
    memset(I, 0, sizeof(double) * k*k);
    for (int i = 0; i < k; i++){
        I[i + k*i] = 1.0;
    }
    
    double *x = (double *)malloc(k*sizeof(double));
    memset(x, 0, sizeof(double) * k);

    double atol = 1e-6;
    int max_iter = 1500;
    int threads = 32;
    int iters = conjugateGradient(I, x, vec, k, atol, max_iter);
    int iters_A = conjugateGradient(ASpd, x_A, res_A, k, atol, max_iter);

    double error = 0.0;
    for (int i = 0; i < k; i++){
        error += sqrt((vec[i] - x[i])*(vec[i] - x[i]));
    }

    double errorA = 0.0;
    for (int i = 0; i < k; i++){
        errorA += sqrt((vec_A[i] - x_A[i]) * (vec_A[i] - x_A[i]));
    }
    printf("CPU:\n");
    printf("I Test: CG Finished with error: %f after %d iterations \n", error, iters);
    printf("A Test: CG Finished with error: %.12f after %d iterations \n", errorA, iters_A);

    memset(x, 0, sizeof(double) * k);
    memset(x_A, 0, sizeof(double) * k);
    atol = 1e-7;
    iters = conjugateGradientCu(I, x, vec, k, atol, max_iter, threads);
    iters_A = conjugateGradientCu(ASpd, x_A, res_A, k, atol, max_iter, threads);

    error = 0.0;
    for (int i = 0; i < k; i++){
        error += sqrt((vec[i] - x[i])*(vec[i] - x[i]));
    }

    errorA = 0.0;
    for (int i = 0; i < k; i++){
        errorA += sqrt((vec_A[i] - x_A[i]) * (vec_A[i] - x_A[i]));
    }
    printf("GPU:\n");
    printf("I Test: CG Finished with error: %f after %d iterations \n", error, iters);
    printf("A Test: CG Finished with error: %.12f after %d iterations \n", errorA, iters_A);

    free(I); I = NULL;
    free(x); x=NULL;
    free(A); free(AT);
    free(ASpd); free(vec_A);
    free(x_A); free(res_A);

}


void test_BLAS_ops(int p, int k, int threadsPerBlock, bool write){

    // Allocate Host Matrices and Vector
    double *A = (double *)malloc(k*k*sizeof(double));

    double *x = (double *)malloc(k*sizeof(double));
    double *b = (double *)malloc(k*sizeof(double));
    double *b_buffer = (double *)malloc(k*sizeof(double)); // to test cuda solutions

    if (A == NULL){
        fprintf(stderr, "Failed to Allocate A\n");
    }

    int numErrors;
    double tol = (double)k * EPS;
    // Initialize timer
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();

    if (k < 1){
        // initialize matrix and vector on cpu
        for (int i = 0; i < k; i++){

            for (int j = 0; j < k; j++){
                A[i] = (double)rand() / (double)RAND_MAX;
            }
            x[i] = (double)rand() / (double)RAND_MAX;
        }
    }
    else{
        // init on GPU
        generate_random_array(k*k, A);
        generate_random_array(k, x);
    }
    
    memset(b, 0, k*sizeof(double));

    // Initialize CUDA Params
    int numBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid_size(numBlocks, 1, 1); // 1D grid
    dim3 block_size(threadsPerBlock, 1, 1);

    dim3 sum_block_size(32, 1, 1);
    dim3 sum_grid_size(1, 1, 1); // 1D grid

    printf("\tNum Blocks = %d for grid size %d\n", numBlocks, k);
    printf("\tNum Threads Launched = %d\n", numBlocks*threadsPerBlock);

    int warps_per_block = (32 + threadsPerBlock - 1) / 32; // every warp in a block needs 32 doubles

    size_t blockMem = 32 * warps_per_block * sizeof(double); // account for each 

    double *b_cu, *A_cu, *x_cu;
    cudaMalloc((void **)&A_cu, k*k*sizeof(double));
    cudaMalloc((void **)&b_cu, k*sizeof(double));
    cudaMalloc((void **)&x_cu, k*sizeof(double));

    cudaMemcpy(A_cu, A, k*k*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_cu, x, k*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(b_cu, 0, k*sizeof(double));

    // T1: Matrix - Vector Product
    //
    //
    printf("#####################################\n");
    printf("Running Test 1: Matrix Vector Product\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Mat Vec %d\n", i);
        timer[i+5] = 0.0;
        t0 = ReadTSC(); // start time
	    matrixVecDense(A, x, b, k, k);
        timer[5+i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    // Warm Up:
    matrixVecCu<<<grid_size, block_size>>>(A_cu, x_cu, b_cu, k, k);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    matrixVecCu<<<grid_size, block_size>>>(A_cu, x_cu, b_cu, k, k);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    char matvec[] = "./output/matrix_vector.csv";
    char matvec_op[] = "MATVEC";
    write_time_to_file(timer, matvec, matvec_op, false, k, write, threadsPerBlock);
    cudaMemcpy(b_buffer, b_cu, k*sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = elementwiseValidate(b_buffer, b, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);

    printf("#####################################\n");
    printf("Running Test 2: Matrix Vector Product OPT\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Mat Vec %d\n", i);
        timer[i+5] = 0.0;
    }

    // Warm Up:
    matrixVecCuOpt<<<grid_size, block_size, blockMem>>>(A_cu, x_cu, b_cu, k, k);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    matrixVecCuOpt<<<grid_size, block_size, blockMem>>>(A_cu, x_cu, b_cu, k, k);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    write_time_to_file(timer, matvec, matvec_op, true, k, false, threadsPerBlock);
    cudaMemcpy(b_buffer, b_cu, k*sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = elementwiseValidate(b_buffer, b, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);

    // T2: Vector Add
    printf("#####################################\n");
    printf("Running Test 3: Vector Add\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Vec + Vec %d\n", i);
        timer[5+i] = 0.0;
        t0 = ReadTSC(); // start time
	    vectorAdd(x, x, b, k);
        timer[5+i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    // Warm Up:
    VecAddCu<<<grid_size, block_size>>>(x_cu, x_cu, b_cu, k);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Vec + Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    VecAddCu<<<grid_size, block_size>>>(x_cu, x_cu, b_cu, k);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    char addvec[] = "./output/vector_add.csv";
    char addvec_op[] = "ADDVEC";
    write_time_to_file(timer, addvec, addvec_op, false, k, write, threadsPerBlock);
    cudaMemcpy(b_buffer, b_cu, k*sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = elementwiseValidate(b_buffer, b, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);

    // T3: Vector Dot

    double r = 0.0;
    double r_buffer[1];
    printf("#####################################\n");
    printf("Running Test 4: Vector Dot Product   \n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Vec x Vec %d\n", i);
        timer[i+5] = 0.0;
        t0 = ReadTSC(); // start time
	    r = vectorDotProduct(x, x, k);
        timer[5+i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    // Warm Up:
    VecDotCu<<<grid_size, block_size>>>(x_cu, x_cu, b_cu, k);
	SumDotCu<<<sum_grid_size, sum_block_size>>>(b_cu, k);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Vec x Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    VecDotCu<<<grid_size, block_size>>>(x_cu, x_cu, b_cu, k);
	    SumDotCu<<<sum_grid_size, sum_block_size>>>(b_cu, k);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    char dotvec[] = "./output/vector_dot.csv";
    char dotvec_op[] = "DOTVEC";
    write_time_to_file(timer, dotvec, dotvec_op, false, k, write,  threadsPerBlock);
    cudaMemcpy(r_buffer, b_cu, sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = fabs(r_buffer[0] - r) > tol ? 1: 0;
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    printf("\t\tAt r=%f, r_cu = %f\n",r,  r_buffer[0]);
    

    printf("#####################################\n");
    printf("Running Test 5: Vector Dot Product OPT\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Vec x Vec %d\n", i);
        timer[5+i] = 0.0; // Set times
    }

    // Warm Up:
    VecDotCuOpt<<<grid_size, block_size, blockMem>>>(x_cu, x_cu, b_cu, k);
    cudaMemset(b_cu, 0, k*sizeof(double));
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        cudaMemset(b_cu, 0, k*sizeof(double));
        timer[i] = 0.0;
        printf("\tRunning GPU Vec x Vec %d\n", i);
        t0 = ReadTSC(); // start time
	    VecDotCuOpt<<<grid_size, block_size, blockMem>>>(x_cu, x_cu, b_cu, k);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    write_time_to_file(timer, dotvec, dotvec_op, true, k, false,  threadsPerBlock);
    cudaMemcpy(r_buffer, b_cu, sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = fabs(r_buffer[0] - r) > tol ? 1: 0;
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    printf("\t\tAt r=%f, r_cu = %f\n",r,  r_buffer[0]);
    
    goto cleanup;


cleanup:
    free(A);
    free(x); free(b);

    cudaFree(A_cu); cudaFree(b_cu); cudaFree(x_cu);

}


void test_MF_ops(int k, int threadsPerBlock, bool write){
    k*=50;
    // Setup domain ish
    float x0 = -1.0f;
    float xN = 1.0f;
    
    int Nx = k-1;
    int bm, bn, bs; 
    int interior;
    double d[7];
    double bd[54];
    double bsd[5];

    
    bm = 1;
    bn = 3;
    bd[0] = 1.0; bd[1] = -2.0; bd[2] = 1.0;
        
    bs = 3;
    bsd[0] = 1.5; bsd[1] = -2.0; bsd[2] = 0.5;
        
    interior = 3;
    d[0] = 1.0; d[1] = -2.0; d[2] = 1.0;
    

    // Allocate Host Matrices and Vector
    double *x = (double *)malloc(k*sizeof(double));
    double *b = (double *)malloc(k*sizeof(double));
    double *b_buffer = (double *)malloc(k*sizeof(double)); // to test cuda solutions


    int numErrors;
    double tol = (double)k * EPS;
    // Initialize timer
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();
    
    memset(b, 0, k*sizeof(double));
    memset(x, 0, k*sizeof(double));
    generate_random_array(k, b);

    double hi = 1.0 / ((xN - x0) / Nx);
    D2_mf *d2_params = create_D2_mf(bm, bn, interior, bs, hi, d, bsd, bd);
    double t  = 1.0;
   

    // Initialize CUDA Params
    int numBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid_size(numBlocks, 1, 1); // 1D grid
    dim3 block_size(threadsPerBlock, 1, 1);

    dim3 sum_block_size(32, 1, 1);
    dim3 sum_grid_size(1, 1, 1); // 1D grid

    printf("\tNum Blocks = %d for grid size %d\n", numBlocks, k);
    printf("\tNum Threads Launched = %d\n", numBlocks*threadsPerBlock);

    int warps_per_block = (32 + threadsPerBlock - 1) / 32; // every warp in a block needs 32 doubles

    size_t blockMem = 32 * warps_per_block * sizeof(double); // account for each 

    double *b_cu, *x_cu;
    cudaMalloc((void **)&b_cu, k*sizeof(double));
    cudaMalloc((void **)&x_cu, k*sizeof(double));

    cudaMemcpy(b_cu, b, k*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(x_cu, 0, k*sizeof(double));

    // T1: Matrix - Vector Product
    printf("#####################################\n");
    printf("Running Test 6: MF Mat Vec \n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Mat Vec %d\n", i);
        timer[i+5] = 0.0;
        t0 = ReadTSC(); // start time
	    D2matrixFreeDir(b, x, k, d2_params);
        timer[5+i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    // Warm Up:
    D2matrixFreeDirCu<<<grid_size, block_size>>>(b_cu, x_cu, k, hi);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    D2matrixFreeDirCu<<<grid_size, block_size>>>(b_cu, x_cu, k, hi);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    char matvec[] = "./output/mf_matvec.csv";
    char matvec_op[] = "STENCIL";
    write_time_to_file(timer, matvec, matvec_op, false, k, write,  threadsPerBlock);
    cudaMemcpy(b_buffer, x_cu, k*sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = elementwiseValidate(b_buffer, x, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);

    printf("#####################################\n");
    printf("Running Test 7: MF Mat Vec Opt\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        printf("\tRunning CPU Mat Vec %d\n", i);
        timer[i+5] = 0.0;
    }

    // Warm Up:
    D2matrixFreeDirCuOpt<<<grid_size, block_size>>>(b_cu, x_cu, k, hi);
    cudaDeviceSynchronize();
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    D2matrixFreeDirCuOpt<<<grid_size, block_size>>>(b_cu, x_cu, k, hi);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    write_time_to_file(timer, matvec, matvec_op, true, k, false,  threadsPerBlock);
    cudaMemcpy(b_buffer, x_cu, k*sizeof(double), cudaMemcpyDeviceToHost);
    numErrors = elementwiseValidate(b_buffer, x, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    
    goto cleanup;


cleanup:
    free(x); free(b);
    freeD2_mf(&d2_params);
    cudaFree(b_cu); cudaFree(x_cu);

}


void test_cg_dense(int k, int threadsPerBlock, bool write){
    
    float x0 = -1.0f;
    float xN =  1.0f;
    
    int Nx = k-1;
    int Nxp = k;
    double mu = 1.0;
    int p = 2;

    float *x = create_1D_domain(x0, xN, Nx);
    double*D1, *H, *HI;
    double *D2, *SAT;
    double *BS;

    double hi = (double)Nx / ((double)(xN-x0));

    printf("Build Operators....");
    create_1D_D1_operator_dense(&D1, &H, &HI, Nx, p, x0, xN);
    create_1D_D2_operator_dense(&D2, &BS, Nx, p, x0, xN);
    // create_H_Mf(&H, &HI, Nx, p, x0, xN);
    addSAT_dir(&SAT, BS, Nx, HI, mu);
    matrixAddDenseIP(D2, SAT, Nxp, Nxp); // Operator with sat is SND

    double *s = (double *)calloc((Nx + 1), sizeof(double));
    double *u = (double *)calloc((Nx + 1), sizeof(double));
    double *b = (double *)calloc(Nxp, sizeof(double)); 
    double *b_spd = (double *)calloc(Nxp, sizeof(double)); 
    double *g = (double *)calloc(Nxp, sizeof(double)); 
    double *u_cu = (double *)calloc(Nxp, sizeof(double)); 
    double *D2_spd = (double *)calloc(Nxp*Nxp, sizeof(double));

    // D2_mf *d2_params = create_D2_mf(bm, bn, interior, bs, hi, d, bsd, bd);
    double t  = 1.0;
    data(g, x, t, Nxp); // Set Data

    memset(b, 0, Nxp*sizeof(double));
    boundary(b, SAT, g, Nxp);
    // boundaryMF(b, d2_params, g, Nxp);
            
    source(s, x, t, Nxp);
    vectorSubIP(b, s, Nxp);

    // make everything SPD
    matrixMulDense(H, D2, D2_spd, Nxp, Nxp, Nxp);
    matrixMulScalarIP(D2_spd, -1.0, Nxp, Nxp); 
    matrixVecDense(H, b, b_spd, Nxp, Nxp);
    vectorMulScalarIP(b_spd, -1.0, Nxp);

    double atol = std::sqrt(1e-16 * vectorDotProduct(b, b, Nxp));
    int max_iter = 10;
            
    printf("Done\n");
    int iters = 0;

    //printf("AFTER CG: b = %.4f\n", vectorDotProduct(b_spd, b_spd, Nxp))
    int numErrors;
    double tol = (double)k * EPS;
    // Initialize timer
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();
   
    // Initialize CUDA Params
    int numBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid_size(numBlocks, 1, 1); // 1D grid
    dim3 block_size(threadsPerBlock, 1, 1);
   
    printf("#####################################\n");
    printf("Running Test 8: CG\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        memset(u, 0, k*sizeof(double)); // reset each time
        printf("\tRunning CPU CG %d\n", i);
        timer[i+5] = 0.0;
        t0 = ReadTSC(); // start time
        iters = conjugateGradient2(D2_spd, u,b_spd, k, atol, max_iter);
        timer[i+5] += ElapsedTime(ReadTSC() - t0); // Set times
        
    }

    // Warm Up:
    iters = conjugateGradientCu(D2_spd, u_cu, b_spd, k, atol, max_iter, threadsPerBlock);
    cudaDeviceSynchronize();
    
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        memset(u_cu, 0, k*sizeof(double));
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    iters = conjugateGradientCu(D2_spd, u_cu, b_spd, k, atol, max_iter, threadsPerBlock);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    char cgfile[] = "./output/cg_dense.csv";
    char cg_op[] = "CGDENSE";
    
    write_time_to_file(timer, cgfile, cg_op, false, k, write,  threadsPerBlock);
    numErrors = elementwiseValidate(u_cu, u, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);

    printf("#####################################\n");
    printf("Running Test 9: CG Opt\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        // memset(u, 0, k*sizeof(double)); // reset each time
        printf("\tRunning CPU CG %d\n", i);
        timer[i+5] = 0.0;
    }

    // Warm Up:
    memset(u_cu, 0, k*sizeof(double));
    iters = conjugateGradientCuOpt(D2_spd, u_cu, b_spd, k, atol, max_iter, threadsPerBlock);
    cudaDeviceSynchronize();
    
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        memset(u_cu, 0, k*sizeof(double));
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    iters = conjugateGradientCuOpt(D2_spd, u_cu, b_spd, k, atol, max_iter, threadsPerBlock);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    write_time_to_file(timer, cgfile, cg_op, true, k, false,  threadsPerBlock);
    numErrors = elementwiseValidate(u_cu, u, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    
    
    goto cleanup;


cleanup:
    free(b);
    free_1D_domain(&x);
    free_D1_ops(&D1, &H, &HI);
    free_D2_ops(&D2, &BS);
    free(b_spd); free(u); free(u_cu);
    free(s); free(g);
    free(SAT);
    free(D2_spd);

}


void test_cg_mf(int k, int threadsPerBlock, bool write){
    
    float x0 = -1.0f;
    float xN =  1.0f;
    
    int Nx = k-1;
    int Nxp = k;
    double mu = 1.0;
    int p = 2;

    p = 2;

    // Add in Support for Other Operators TO DO
    int bm, bn, bs; 
    int interior;
    double d[3];
    double bd[3];
    double bsd[3];

    
    bm = 1;
    bn = 3;
    bd[0] = 1.0; bd[1] = -2.0; bd[2] = 1.0;
        
    bs = 3;
    bsd[0] = 1.5; bsd[1] = -2.0; bsd[2] = 0.5;
        
    interior = 3;
    d[0] = 1.0; d[1] = -2.0; d[2] = 1.0;

    float *x = create_1D_domain(x0, xN, Nx);
    double*H, *HI;
    

    // create_H_Mf(&H, &HI, Nx, p, x0, xN);
    double hi = (double)Nx / ((double)(xN-x0));
    D2_mf *d2_params = create_D2_mf(bm, bn, interior, bs, hi, d, bsd, bd);

    printf("Build Operators....");
    
    // create_H_Mf(&H, &HI, Nx, p, x0, xN);

    double *s = (double *)calloc((Nx + 1), sizeof(double));
    double *u = (double *)calloc((Nx + 1), sizeof(double));
    double *b = (double *)calloc(Nxp, sizeof(double)); 
    double *b_spd = (double *)calloc(Nxp, sizeof(double)); 
    double *g = (double *)calloc(Nxp, sizeof(double)); 
    double *u_cu = (double *)calloc(Nxp, sizeof(double)); 
    

    
    double t  = 1.0;
    data(g, x, t, Nxp); // Set Data

    boundaryMF(b, d2_params, g, Nxp);
            
    source(s, x, t, Nxp);
    //vectorMulEW(s, H, s, Nxp);

    vectorMulScalarIP(s, (1/hi)*-1.0, Nxp);
    s[0] *= 2;
    s[k-1] *= 2;
    vectorSubIP(b, s, Nxp);

    double atol = std::sqrt(1e-16 * vectorDotProduct(b, b, Nxp));
    int max_iter = 10;
            
    printf("Done\n");
    int iters = 0;

    int numErrors;
    double tol = (double)k * EPS;
    // Initialize timer
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();
   
    // Initialize CUDA Params
    int numBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid_size(numBlocks, 1, 1); // 1D grid
    dim3 block_size(threadsPerBlock, 1, 1);
    int iters_cu;
    printf("#####################################\n");
    printf("Running Test 10: CG MF\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        memset(u, 0, k*sizeof(double)); // reset each time
        printf("\tRunning CPU CG %d\n", i);
        timer[i+5] = 0.0;
        t0 = ReadTSC(); // start time
        iters = conjugateGradientMFCpu(u, b, k, atol, max_iter, d2_params);
        timer[i+5] += ElapsedTime(ReadTSC() - t0); // Set times
        
    }

    // Warm Up:
    iters_cu = conjugateGradientCuMF(u_cu, b, hi, k, atol, max_iter, threadsPerBlock);
    cudaDeviceSynchronize();
    
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        memset(u_cu, 0, k*sizeof(double));
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    iters_cu = conjugateGradientCuMF(u_cu, b, hi, k, atol, max_iter, threadsPerBlock);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }

    char cgfile[] = "./output/cg_mf.csv";
    char cg_op[] = "CGMF";
    
    write_time_to_file(timer, cgfile, cg_op, false, k, write,  threadsPerBlock);
    numErrors = elementwiseValidate(u_cu, u, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    printf("\t\tAt k=%d, cpu takes %d iters gpu takes %d iters\n", k, iters, iters_cu);

    printf("#####################################\n");
    printf("Running Test 11: CG MF Opt\n");
    printf("#####################################\n");
	for (int i =0; i < 5; i++){
        // memset(u, 0, k*sizeof(double)); // reset each time
        printf("\tRunning CPU CG %d\n", i);
        timer[i+5] = 0.0;
    }

    // Warm Up:
    memset(u_cu, 0, k*sizeof(double));
    iters = conjugateGradientCuMFOpt(u_cu, b, hi, k, atol, max_iter, threadsPerBlock);
    cudaDeviceSynchronize();
    
    // T1: Matrix - Vector Product
	for (int i =0; i < 5; i++){
        memset(u_cu, 0, k*sizeof(double));
        printf("\tRunning GPU Mat Vec %d\n", i);
        timer[i] = 0.0;
        t0 = ReadTSC(); // start time
	    iters = conjugateGradientCuMFOpt(u_cu, b, hi, k, atol, max_iter, threadsPerBlock);
        cudaDeviceSynchronize();
        timer[i] += ElapsedTime(ReadTSC() - t0); // Set times
    }
    
    write_time_to_file(timer, cgfile, cg_op, true, k, false,  threadsPerBlock);
    numErrors = elementwiseValidate(u_cu, u, k, tol);
    printf("\t\tAt k=%d, found %d errors with tol %f\n", k, numErrors, tol);
    
    
    goto cleanup;


cleanup:
    free(b);
    free_1D_domain(&x);
    // free(H); free(HI); 
    free(b_spd); free(u); free(u_cu);
    free(s); free(g);
    free(d2_params);



}







int elementwiseValidate(double *a, double *b, int k, double tol){
    int numErrors = 0;
    for (int i=0; i<k; i++){
        if (fabs(a[i] - b[i]) > tol){
            numErrors++;
        }
    }
    return numErrors;
}
/* Print timing information 
 * Remaining Code is for Printing info and timing (Mostly from CS531)
 */
void write_time_to_file(double timer[], char *filename, char* operation, bool Opt, int k, bool overwrite, int threads)
{
    FILE *fptr;
    if (overwrite){
        fptr = fopen(filename, "w");
        fprintf(fptr, "Device,Operation,Time(s),Opt,grid_size,Run,Threads\n");
    }
        
    else{
        fptr = fopen(filename, "a");
    }

    if (fptr == NULL){
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < 5; i++){
        fprintf(fptr, "GPU,%s,%.8f,%d,%d,%d,%d\n", operation, timer[i], Opt, k, i,threads);
    }
    for (int i = 0; i < 5; i++){
        fprintf(fptr, "CPU,%s,%.8f,%d,%d,%d,1\n", operation, timer[i+5], Opt, k, i);
    }


    

    fclose(fptr);

}


/*
 *
 * Util to make big random arrays
 *
 */

 // Kernel to initialize the random number generator states
__global__ void setup_kernel(curandState *state, unsigned long long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    /* Initialize a state for each thread, using the thread ID as a sequence number */
    if (id < n){
        curand_init(seed, id, 0, &state[id]);
    }
    
}

// Kernel to generate random numbers and store them in an array
__global__ void generate_kernel(curandState *state, double *rand_nums, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        /* Generate a uniformly distributed float between 0.0 and 1.0 */
        rand_nums[id] = (double)curand_uniform(&state[id]); // cast to double
    }
}

void generate_random_array(int n, double *A) {
   
    // Allocate memory on the host and device
    curandState *d_state;
    double *d_rand_nums;
    cudaMalloc((void **)&d_state, n *  sizeof(curandState));
    cudaMalloc((void **)&d_rand_nums, n * sizeof(double));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    setup_kernel<<<blocks, threadsPerBlock>>>(d_state, 1234ULL, n);
    generate_kernel<<<blocks, threadsPerBlock>>>(d_state, d_rand_nums, n);

    cudaMemcpy(A, d_rand_nums, n*sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_state);
    cudaFree(d_rand_nums);
}