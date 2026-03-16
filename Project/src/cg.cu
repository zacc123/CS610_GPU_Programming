#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "../include/cg.cuh"
#include "../include/matrix_ops.cuh"
#include "../include/utils.h"


int conjugateGradient(double *A, double *x, double *b, int k, double atol, int max_iter){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *r_0, *r_1, *p_0, *p_1, *tmp_res, *tmp_res2, *x_k, *x_k1;
	double *pointers[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
	double clamp_tol = 1e-16;
	double atol2 = atol * atol;


	size_t array_size = k * sizeof(double);
	for (int i = 0; i < 8; i ++){
		pointers[i] = (double *)malloc(array_size);
		if (pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			exit(EXIT_FAILURE);
		}
		memset(pointers[i], 0, array_size);
	}
	r_0     = pointers[0];
    r_1     = pointers[1];
    p_0     = pointers[2];
    p_1     = pointers[3];
    tmp_res = pointers[4];
    tmp_res2= pointers[5];
    x_k     = pointers[6];
    x_k1    = pointers[7];

	// compute initial residual
	matrixVecDense(A, x, tmp_res, k, k);
	vectorSub(b, tmp_res, r_0, k);
	double residual2 = vectorDotProduct(r_0, r_0, k);
	
	// Check If low enough
	// if yes, then x0 was close enough
	if (residual2 <= atol2){
		return 0;
	}

	memcpy(p_0, r_0, array_size);
	memcpy(x_k, x, array_size);

	// Keep track of total iterations 
	int iter = 1;

	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;

	for (int i = 0; i < max_iter; i++){

		memset(tmp_res, 0, array_size); // clear before this part
		memset(tmp_res2, 0, array_size); // clear before this part

		alpha_k = vectorDotProduct(r_0, r_0, k);
		matrixVecDense(A, p_0, tmp_res, k, k);
		alpha_k = alpha_k / clamp(vectorDotProduct(p_0, tmp_res, k), clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)

		memset(tmp_res, 0, array_size); // clear before this part
		vectorMulScalar(p_0, alpha_k, tmp_res, k);
		vectorAdd(x_k, tmp_res, x_k1, k); // x_k1 = x_k + alpha pk

		memset(tmp_res, 0, array_size); // clear before this part
		matrixVecDense(A, p_0, tmp_res, k, k);
		vectorMulScalar(tmp_res, alpha_k, tmp_res2, k);
		vectorSub(r_0, tmp_res2, r_1, k);

		residual2 = vectorDotProduct(r_1, r_1, k);
		if (residual2 <= atol2){
			//printf("Error is %.16f\n", residual2);
			break;
		}

		beta_k = residual2 / clamp(vectorDotProduct(r_0, r_0, k), clamp_tol);
		memset(tmp_res, 0, array_size); // clear before this part
		vectorMulScalar(p_0, beta_k, tmp_res, k);
		vectorAdd(r_1, tmp_res, p_1, k);

		// Now copy everything over
		memcpy(p_0, p_1, array_size);
		memcpy(r_0, r_1, array_size);
		memcpy(x_k, x_k1, array_size);
		iter++;
	}

	// Finally, copy x_k1 to x and done
	memcpy(x, x_k1, array_size);

	// cleanup
	for (int i = 0; i < 8; i++){
		free(pointers[i]); pointers[i] = NULL;
	}
	return iter;
}

int conjugateGradient2(double *A, double *x, double *b, int k, double atol, int max_iter){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *r, *p0, *tmp;
	double *pointers[3] = {NULL,NULL,NULL};
	double clamp_tol = 1e-16;
	double atol2 = atol * atol; // Square the tolerance to avoid taking a sqrt

	size_t array_size = k * sizeof(double);
	for (int i = 0; i < 3; i ++){
		pointers[i] = (double *)malloc(array_size);
		if (pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			exit(EXIT_FAILURE);
		}
		memset(pointers[i], 0, array_size);
	}

	// Assign buffers
	r       = pointers[0];
    p0      = pointers[1];
    tmp     = pointers[2];

	// compute initial residual
	matrixVecDense(A, x, tmp, k, k);
	vectorSub(b, tmp, r, k);
	double residual2 = vectorDotProduct(r, r, k);
	

	int iter = 0;
	// Check If low enough
	// if yes, then x0 was close enough
	if (residual2 <= atol2){
		goto cleanup;
	}
	// printf("At iter %u, Residual = %.16f\n", iter, residual2);
	memcpy(p0, r, array_size);

	// Keep track of total iterations 
	iter = 1;

	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;
	double r0;

	for (int i = 0; i < max_iter; i++){

		r0 = vectorDotProduct(r, r, k); // rk' rk
		matrixVecDense(A, p0, tmp, k, k);
		alpha_k = r0 / clamp(vectorDotProduct(p0, tmp, k), clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)

		vectorMulScalar(p0, alpha_k, tmp, k);
		vectorAdd(x, tmp, x, k); // x_k1 = x_k + alpha pk

		matrixVecDense(A, p0, tmp, k, k);
		vectorMulScalar(tmp, alpha_k, tmp, k);
		vectorSub(r, tmp, r, k); // r is now rk+1

		residual2 = vectorDotProduct(r, r, k);
		if (residual2 <= atol2){
			// printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		beta_k = residual2 / clamp(r0, clamp_tol);
		vectorMulScalar(p0, beta_k, p0, k);
		vectorAdd(r, p0, p0, k);

		// Now copy everything over
		iter++;
	}

	// Cleaning
	cleanup:
	for (int i = 0; i < 3; i++){
		free(pointers[i]); pointers[i] = NULL;
	}
	return iter;
}

int conjugateGradientMFCpu(double *x, double *b, int k, double atol, int max_iter, D2_mf *d2_params){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *r, *p0, *tmp;
	double *pointers[3] = {NULL,NULL,NULL};
	double clamp_tol = 1e-16;
	double atol2 = atol * atol; // Square the tolerance to avoid taking a sqrt

	size_t array_size = k * sizeof(double);
	for (int i = 0; i < 3; i ++){
		pointers[i] = (double *)malloc(array_size);
		if (pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			exit(EXIT_FAILURE);
		}
		memset(pointers[i], 0, array_size);
	}

	// Assign buffers
	r       = pointers[0];
    p0      = pointers[1];
    tmp     = pointers[2];

	// compute initial residual
	D2matrixFreeDir(x, tmp, k, d2_params);
	vectorSub(b, tmp, r, k);
	double residual2 = vectorDotProduct(r, r, k);
	

	int iter = 0;
	// Check If low enough
	// if yes, then x0 was close enough
	if (residual2 <= atol2){
		goto cleanup;
	}
	//printf("At iter %u, Residual = %.16f\n", iter, residual2);
	memcpy(p0, r, array_size);

	// Keep track of total iterations 
	iter = 1;

	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;
	double r0;

	for (int i = 0; i < max_iter; i++){

		r0 = vectorDotProduct(r, r, k); // rk' rk
		D2matrixFreeDir(p0, tmp, k, d2_params);
		alpha_k = r0 / clamp(vectorDotProduct(p0, tmp, k), clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)

		vectorMulScalar(p0, alpha_k, tmp, k);
		vectorAdd(x, tmp, x, k); // x_k1 = x_k + alpha pk

		D2matrixFreeDir(p0, tmp, k, d2_params);
		vectorMulScalar(tmp, alpha_k, tmp, k);
		vectorSub(r, tmp, r, k); // r is now rk+1

		residual2 = vectorDotProduct(r, r, k);
		if (residual2 <= atol2){
			// printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		beta_k = residual2 / clamp(r0, clamp_tol);
		vectorMulScalar(p0, beta_k, p0, k);
		vectorAdd(r, p0, p0, k);

		// Now copy everything over
		iter++;
	}

	// Cleaning
	cleanup:
	for (int i = 0; i < 3; i++){
		free(pointers[i]); pointers[i] = NULL;
	}
	return iter;
}



int conjugateGradientCu(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *d_r, *d_p0, *d_x, *d_tmp;
	double *device_pointers[9] = {NULL,NULL,NULL,NULL};

	double *h_r;
	double *host_pointers[2] = {NULL, NULL};

	double clamp_tol = 1e-16;
	double atol2 = atol * atol;

	// Do Memory Allocation

	double *dA;
	size_t matrix_size = k * k * sizeof(double);

	int numBlocksX = (int)((k +  threadsPerBlock - 1) / threadsPerBlock);
	// printf("NumBlocks = %d\n", numBlocksX);
	dim3 blocks(numBlocksX, 1, 1);
	dim3 threads(threadsPerBlock, 1, 1);

	dim3 sumBlocks(1, 1, 1);
	dim3 sumThreads(32, 1, 1);

	double residual2 = 0.0;
	double residual0 = 0.0;

	int iter = 0;

	// printf("B4 MEM CG: b = %.4f", vectorDotProduct(b, b, k));
	bool fail = false;
	// Host
	size_t array_size = k * sizeof(double);

	size_t r_size = sizeof(double); // tweaked without testing whooooooops
	for (int i = 0; i < 1; i ++){
		host_pointers[i] = (double *)malloc(array_size);
		if (host_pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(host_pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			fail = true;
		}
		memset(host_pointers[i], 0, array_size);
	}


	// device
	for (int i = 0; i < 4; i ++){
		cudaMalloc((void **)&device_pointers[i],array_size); 
		cudaMemset(device_pointers[i], 0, array_size);
	}
	// Also need A Array over there
	
	cudaMalloc((void **)&dA, matrix_size);
	cudaMemset(dA, 0, matrix_size);

	d_r    = device_pointers[0];
    d_p0    = device_pointers[1];
    d_x    = device_pointers[2];
    d_tmp   = device_pointers[3];

	h_r = host_pointers[0];
    
	// printf("B4 MEM CPY CG: b = %.4f", vectorDotProduct(b, b, k));
	cudaMemcpy(dA, A, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, b, array_size, cudaMemcpyHostToDevice);
	
	
	// compute initial residual
	matrixVecCu<<<blocks, threads>>>(dA, d_x, d_p0, k, k); // A bit confusing at first but will make more sense
	VecSubCu<<<blocks, threads>>>(d_tmp, d_p0, d_r, k);;
	VecDotCu<<<blocks, threads>>>(d_r, d_r, d_tmp, k);
	SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
	cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
	// printf("At iter %u, Residual = %.16f\n", iter, h_r[0]);
	
	
	
	// Check If low enough
	// if yes, then x0 was close enough
	residual2 = h_r[0];
	residual0 = h_r[0];
	// printf("At iter %u, Residual = %.16f\n", iter, residual2);
	if (residual2 <= atol2){
		goto cleanup;
	}

	cudaMemcpy(d_p0, d_r, array_size, cudaMemcpyDeviceToDevice);
	
	// Keep track of total iterations 
	iter = 1;
	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;

	if (fail){
		goto cleanup;
	}

	for (int i = 0; i < max_iter; i++){

		// DAG 4
		// VecDotCu<<<blocks, threads>>>(d_r0, d_r0, d_dotbuffer, k);
		
		
		matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);

		VecDotCu<<<blocks, threads>>>(d_p0, d_tmp, d_tmp, k);
		SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
		cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
		alpha_k = residual0 / clamp(h_r[0], clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)

		// DAG 5
		VecMulScalarCu<<<blocks, threads>>>(d_p0, d_tmp, alpha_k, k);
		VecAddCu<<<blocks, threads>>>(d_x, d_tmp, d_x, k); // x_k1 = x_k + alpha pk

		// DAG 6
		matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);
		VecMulScalarIPCu<<<blocks, threads>>>(d_tmp, alpha_k, k);
		VecSubCu<<<blocks, threads>>>(d_r, d_tmp, d_r, k);
		VecDotCu<<<blocks, threads>>>(d_r, d_r, d_tmp, k);
		SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
		cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
		
		//DAG 7
		residual2 = h_r[0];
		if (residual2 > 100000.0 || residual2 < 0){
			printf("Blew up");
			goto cleanup;
		}
		if (residual2 <= atol2){
			//printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		// DAG 8 ... noice
		beta_k = residual2 / clamp(residual0, clamp_tol);
		
		// DAG 9
		VecMulScalarIPCu<<<blocks, threads>>>(d_p0, beta_k, k);
		VecAddCu<<<blocks, threads>>>(d_r, d_p0, d_p0, k);
		residual0 =  residual2;
		iter++;
		cudaDeviceSynchronize();
	}

cleanup:
	// Finally, copy x_k1 to x and done

	if (!fail){
		cudaMemcpy(x, d_x, array_size, cudaMemcpyDeviceToHost);
	}
	

	// Do Memory Allocation cleaning

	// Host
	for (int i = 0; i < 1; i ++){
		free(host_pointers[i]);
		host_pointers[i] = NULL;
	}


	// device
	
	for (int i = 0; i < 4; i ++){
		cudaFree(device_pointers[i]);
		device_pointers[i] = NULL;
	}

	cudaFree(dA);
	return iter;
}



int conjugateGradientCuMF(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *d_r, *d_p0, *d_x, *d_tmp;
	double *device_pointers[4] = {NULL,NULL,NULL,NULL};

	double *h_r;
	double *host_pointers[2] = {NULL, NULL};

	double clamp_tol = 1e-16;
	double atol2 = atol * atol;

	// Do Memory Allocation
	int numBlocksX = (int)((k +  threadsPerBlock - 1) / threadsPerBlock);
	dim3 blocks(numBlocksX, 1, 1);
	dim3 threads(threadsPerBlock, 1, 1);

	dim3 sumBlocks(1, 1, 1);
	dim3 sumThreads(32, 1, 1);

	double residual2 = 0.0;
	double residual0 = 0.0;

	int iter = 0;

	// printf("B4 MEM CG: b = %.4f", vectorDotProduct(b, b, k));
	bool fail = false;
	// Host
	size_t array_size = k * sizeof(double);

	size_t r_size = k * sizeof(double);
	for (int i = 0; i < 1; i ++){
		host_pointers[i] = (double *)malloc(array_size);
		if (host_pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(host_pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			fail = true;
		}
		memset(host_pointers[i], 0, array_size);
	}


	// device
	for (int i = 0; i < 4; i ++){
		cudaMalloc((void **)&device_pointers[i],array_size); 
		cudaMemset(device_pointers[i], 0, array_size);
	}
	// Also need A Array over there
	d_r    = device_pointers[0];
    d_p0    = device_pointers[1];
    d_x    = device_pointers[2];
    d_tmp   = device_pointers[3];

	h_r = host_pointers[0];

	// Bring over the mf params
	// printf("B4 MEM CPY CG: b = %.4f", vectorDotProduct(b, b, k));
	cudaMemcpy(d_x, x, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, b, array_size, cudaMemcpyHostToDevice);
	
	
	// compute initial residual
	//matrixVecCu<<<blocks, threads>>>(dA, d_x, d_p0, k, k); // A bit confusing at first but will make more sense
	D2matrixFreeDirCu<<<blocks, threads>>>(d_x, d_p0, k, hi);
	VecSubCu<<<blocks, threads>>>(d_tmp, d_p0, d_r, k);
	VecDotCu<<<blocks, threads>>>(d_r, d_r, d_tmp, k);
	SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
	cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
	// printf("At iter %u, Residual = %.16f\n", iter, h_r[0]);
	
	
	
	// Check If low enough
	// if yes, then x0 was close enough
	residual2 = h_r[0];
	residual0 = h_r[0];
	//printf("At iter %u, Residual = %.16f\n", iter, residual2);
	if (residual2 <= atol2){
		goto cleanup;
	}

	cudaMemcpy(d_p0, d_r, array_size, cudaMemcpyDeviceToDevice);
	
	// Keep track of total iterations 
	iter = 1;
	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;

	if (fail){
		goto cleanup;
	}

	for (int i = 0; i < max_iter; i++){

		// DAG 4
		// VecDotCu<<<blocks, threads>>>(d_r0, d_r0, d_dotbuffer, k);
		
		D2matrixFreeDirCu<<<blocks, threads>>>(d_p0, d_tmp, k, hi);
		// matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);

		VecDotCu<<<blocks, threads>>>(d_p0, d_tmp, d_tmp, k);
		SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
		cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
		//printf("[2] At iter %u, Residual = %.16f\n", iter, h_r[0]);
		alpha_k = residual0 / clamp(h_r[0], clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)
		//printf("[3] At iter %u, Residual = %.16f\n", iter, alpha_k);
		// DAG 5
		VecMulScalarCu<<<blocks, threads>>>(d_p0, d_tmp, alpha_k, k);
		VecAddCu<<<blocks, threads>>>(d_x, d_tmp, d_x, k); // x_k1 = x_k + alpha pk

		// DAG 6
		//matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);
		D2matrixFreeDirCu<<<blocks, threads>>>(d_p0, d_tmp, k, hi);
		VecMulScalarIPCu<<<blocks, threads>>>(d_tmp, alpha_k, k);
		VecSubCu<<<blocks, threads>>>(d_r, d_tmp, d_r, k);
		VecDotCu<<<blocks, threads>>>(d_r, d_r, d_tmp, k);
		SumDotCu<<<sumBlocks, sumThreads>>>(d_tmp, k);
		cudaMemcpy(h_r, d_tmp, r_size, cudaMemcpyDeviceToHost);
		
		//DAG 7
		residual2 = h_r[0];
		if (residual2 > 100000.0 || residual2 < 0){
			printf("Blew up");
			goto cleanup;
		}
		if (residual2 <= atol2){
			//printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		// DAG 8 ... noice
		beta_k = residual2 / clamp(residual0, clamp_tol);
		
		// DAG 9
		VecMulScalarIPCu<<<blocks, threads>>>(d_p0, beta_k, k);
		VecAddCu<<<blocks, threads>>>(d_r, d_p0, d_p0, k);
		residual0 =  residual2;
		iter++;
		cudaDeviceSynchronize();
	}

cleanup:
	// Finally, copy x_k1 to x and done

	if (!fail){
		cudaMemcpy(x, d_x, array_size, cudaMemcpyDeviceToHost);
	}
	

	// Do Memory Allocation cleaning

	// Host
	for (int i = 0; i < 1; i ++){
		free(host_pointers[i]);
		host_pointers[i] = NULL;
	}


	// device
	
	for (int i = 0; i < 4; i ++){
		cudaFree(device_pointers[i]);
		device_pointers[i] = NULL;
	}

	return iter;
}



int conjugateGradientCuOpt(double *A, double *x, double *b, int k, double atol, int max_iter, int threadsPerBlock){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *d_r, *d_p0, *d_x, *d_tmp, *d_dot;
	double *device_pointers[5] = {NULL,NULL,NULL,NULL, NULL};

	double *h_r;
	double *host_pointers[2] = {NULL, NULL};

	double clamp_tol = 1e-16;
	double atol2 = atol * atol;

	// Do Memory Allocation

	double *dA;
	size_t matrix_size = k * k * sizeof(double);

	int numBlocksX = (int)((k +  threadsPerBlock - 1) / threadsPerBlock);
	// printf("NumBlocks = %d\n", numBlocksX);
	dim3 blocks(numBlocksX, 1, 1);
	dim3 threads(threadsPerBlock, 1, 1);

	dim3 sumBlocks(1, 1, 1);
	dim3 sumThreads(32, 1, 1);

	double residual2 = 0.0;
	double residual0 = 0.0;

	int iter = 0;

	int warps_per_block = (32 + threadsPerBlock - 1) / 32; // every warp in a block needs 32 doubles

    size_t blockMem = 32 *  warps_per_block * sizeof(double); // account for each 
	// printf("B4 MEM CG: b = %.4f", vectorDotProduct(b, b, k));
	bool fail = false;
	// Host
	size_t array_size = k * sizeof(double);

	size_t r_size = sizeof(double); // tweaked without testing whooooooops
	for (int i = 0; i < 1; i ++){
		host_pointers[i] = (double *)malloc(array_size);
		if (host_pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(host_pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			fail = true;
		}
		memset(host_pointers[i], 0, array_size);
	}


	// device
	for (int i = 0; i < 5; i ++){
		cudaMalloc((void **)&device_pointers[i],array_size); 
		cudaMemset(device_pointers[i], 0, array_size);
	}
	// Also need A Array over there
	
	cudaMalloc((void **)&dA, matrix_size);
	cudaMemset(dA, 0, matrix_size);

	d_r    = device_pointers[0];
    d_p0    = device_pointers[1];
    d_x    = device_pointers[2];
    d_tmp   = device_pointers[3];
	d_dot   = device_pointers[4];

	h_r = host_pointers[0];
    
	// printf("B4 MEM CPY CG: b = %.4f", vectorDotProduct(b, b, k));
	cudaMemcpy(dA, A, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, b, array_size, cudaMemcpyHostToDevice);
	
	
	// compute initial residual
	matrixVecCuOpt<<<blocks, threads, blockMem>>>(dA, d_x, d_p0, k, k); // A bit confusing at first but will make more sense
	VecSubCu<<<blocks, threads>>>(d_tmp, d_p0, d_r, k);
	cudaMemset(d_dot, 0, sizeof(double));
	VecDotCuOpt<<<blocks, threads, blockMem>>>(d_r, d_r, d_dot, k);
	cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
	// printf("At iter %u, Residual = %.16f\n", iter, h_r[0]);
	
	// Check If low enough
	// if yes, then x0 was close enough
	residual2 = h_r[0];
	residual0 = h_r[0];
	// printf("At iter %u, Residual = %.16f\n", iter, residual2);
	if (residual2 <= atol2){
		goto cleanup;
	}

	cudaMemcpy(d_p0, d_r, array_size, cudaMemcpyDeviceToDevice);
	
	// Keep track of total iterations 
	iter = 1;
	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;

	if (fail){
		goto cleanup;
	}

	for (int i = 0; i < max_iter; i++){

		// DAG 4
		// VecDotCu<<<blocks, threads>>>(d_r0, d_r0, d_dotbuffer, k);
		
		
		matrixVecCuOpt<<<blocks, threads, blockMem>>>(dA, d_p0, d_tmp, k, k);
		cudaMemset(d_dot, 0, sizeof(double));
		VecDotCuOpt<<<blocks, threads, blockMem>>>(d_p0, d_tmp, d_dot, k);
		cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
		alpha_k = residual0 / clamp(h_r[0], clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)

		// DAG 5
		VecMulScalarCu<<<blocks, threads>>>(d_p0, d_tmp, alpha_k, k);
		VecAddCu<<<blocks, threads>>>(d_x, d_tmp, d_x, k); // x_k1 = x_k + alpha pk

		// DAG 6
		matrixVecCuOpt<<<blocks, threads, blockMem>>>(dA, d_p0, d_tmp, k, k);
		VecMulScalarIPCu<<<blocks, threads>>>(d_tmp, alpha_k, k);
		VecSubCu<<<blocks, threads>>>(d_r, d_tmp, d_r, k);
		cudaMemset(d_dot, 0, sizeof(double));
		VecDotCuOpt<<<blocks, threads, blockMem>>>(d_r, d_r, d_dot, k);
		cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
		
		//DAG 7
		residual2 = h_r[0];
		if (residual2 > 100000.0 || residual2 < 0){
			printf("Blew up");
			goto cleanup;
		}
		if (residual2 <= atol2){
			//printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		// DAG 8 ... noice
		beta_k = residual2 / clamp(residual0, clamp_tol);
		
		// DAG 9
		VecMulScalarIPCu<<<blocks, threads>>>(d_p0, beta_k, k);
		VecAddCu<<<blocks, threads>>>(d_r, d_p0, d_p0, k);
		residual0 =  residual2;
		iter++;
		cudaDeviceSynchronize();
	}

cleanup:
	// Finally, copy x_k1 to x and done

	if (!fail){
		cudaMemcpy(x, d_x, array_size, cudaMemcpyDeviceToHost);
	}
	

	// Do Memory Allocation cleaning

	// Host
	for (int i = 0; i < 1; i ++){
		free(host_pointers[i]);
		host_pointers[i] = NULL;
	}


	// device
	
	for (int i = 0; i < 5; i ++){
		cudaFree(device_pointers[i]);
		device_pointers[i] = NULL;
	}

	cudaFree(dA);
	return iter;
}

int conjugateGradientCuMFOpt(double *x, double *b, double hi, int k, double atol, int max_iter, int threadsPerBlock){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *d_r, *d_p0, *d_x, *d_tmp, *d_dot;
	double *device_pointers[5] = {NULL,NULL,NULL,NULL, NULL};

	double *h_r;
	double *host_pointers[2] = {NULL, NULL};

	double clamp_tol = 1e-16;
	double atol2 = atol * atol;

	// Do Memory Allocation
	int numBlocksX = (int)((k +  threadsPerBlock - 1) / threadsPerBlock);
	dim3 blocks(numBlocksX, 1, 1);
	dim3 threads(threadsPerBlock, 1, 1);

	dim3 sumBlocks(1, 1, 1);
	dim3 sumThreads(32, 1, 1);

	double residual2 = 0.0;
	double residual0 = 0.0;

	int iter = 0;

	int warps_per_block = (32 + threadsPerBlock - 1) / 32; // every warp in a block needs 32 doubles

    size_t blockMem = 32 * warps_per_block * sizeof(double); // account for each 
	// printf("B4 MEM CG: b = %.4f", vectorDotProduct(b, b, k));
	bool fail = false;
	// Host
	size_t array_size = k * sizeof(double);

	size_t r_size = k * sizeof(double);
	for (int i = 0; i < 1; i ++){
		host_pointers[i] = (double *)malloc(array_size);
		if (host_pointers[i] == NULL){
			for (int j = 0; j < i; j++) free(host_pointers[j]);
			fprintf(stderr, "Failed to Allocated Memory in CG");
			fail = true;
		}
		memset(host_pointers[i], 0, array_size);
	}


	// device
	for (int i = 0; i < 5; i ++){
		cudaMalloc((void **)&device_pointers[i],array_size); 
		cudaMemset(device_pointers[i], 0, array_size);
	}
	// Also need A Array over there
	d_r    = device_pointers[0];
    d_p0    = device_pointers[1];
    d_x    = device_pointers[2];
    d_tmp   = device_pointers[3];
	d_dot   = device_pointers[4];

	h_r = host_pointers[0];

	// Bring over the mf params
	// printf("B4 MEM CPY CG: b = %.4f", vectorDotProduct(b, b, k));
	cudaMemcpy(d_x, x, array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, b, array_size, cudaMemcpyHostToDevice);
	
	
	// compute initial residual
	//matrixVecCu<<<blocks, threads>>>(dA, d_x, d_p0, k, k); // A bit confusing at first but will make more sense
	D2matrixFreeDirCu<<<blocks, threads>>>(d_x, d_p0, k, hi);
	VecSubCu<<<blocks, threads>>>(d_tmp, d_p0, d_r, k);
	cudaMemset(d_dot, 0, sizeof(double));
	VecDotCuOpt<<<blocks, threads, blockMem>>>(d_r, d_r, d_dot, k);
	cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
	// printf("At iter %u, Residual = %.16f\n", iter, h_r[0]);
	
	
	
	// Check If low enough
	// if yes, then x0 was close enough
	residual2 = h_r[0];
	residual0 = h_r[0];
	//printf("At iter %u, Residual = %.16f\n", iter, residual2);
	if (residual2 <= atol2){
		goto cleanup;
	}

	cudaMemcpy(d_p0, d_r, array_size, cudaMemcpyDeviceToDevice);
	
	// Keep track of total iterations 
	iter = 1;
	// Actual CG loop
	// p0 -> pk, p1 -> pk+1
	double alpha_k;
	double beta_k;

	if (fail){
		goto cleanup;
	}

	for (int i = 0; i < max_iter; i++){

		// DAG 4
		// VecDotCu<<<blocks, threads>>>(d_r0, d_r0, d_dotbuffer, k);
		
		D2matrixFreeDirCu<<<blocks, threads>>>(d_p0, d_tmp, k, hi);
		// matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);
		cudaMemset(d_dot, 0, sizeof(double));
		VecDotCuOpt<<<blocks, threads, blockMem>>>(d_p0, d_tmp, d_dot, k);
		cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
		//printf("[2] At iter %u, Residual = %.16f\n", iter, h_r[0]);
		alpha_k = residual0 / clamp(h_r[0], clamp_tol);  //. alpha = (rk'rk) / (pk'Apk)
		//printf("[3] At iter %u, Residual = %.16f\n", iter, alpha_k);
		// DAG 5
		VecMulScalarCu<<<blocks, threads>>>(d_p0, d_tmp, alpha_k, k);
		VecAddCu<<<blocks, threads>>>(d_x, d_tmp, d_x, k); // x_k1 = x_k + alpha pk

		// DAG 6
		//matrixVecCu<<<blocks, threads>>>(dA, d_p0, d_tmp, k, k);
		D2matrixFreeDirCu<<<blocks, threads>>>(d_p0, d_tmp, k, hi);
		VecMulScalarIPCu<<<blocks, threads>>>(d_tmp, alpha_k, k);
		VecSubCu<<<blocks, threads>>>(d_r, d_tmp, d_r, k);
		cudaMemset(d_dot, 0, sizeof(double));
		VecDotCu<<<blocks, threads>>>(d_r, d_r, d_dot, k);
		cudaMemcpy(h_r, d_dot, r_size, cudaMemcpyDeviceToHost);
		
		//DAG 7
		residual2 = h_r[0];
		if (residual2 > 100000.0 || residual2 < 0){
			printf("Blew up");
			goto cleanup;
		}
		if (residual2 <= atol2){
			//printf("Error is %.16f\n", residual2);
			goto cleanup;
		}

		// DAG 8 ... noice
		beta_k = residual2 / clamp(residual0, clamp_tol);
		
		// DAG 9
		VecMulScalarIPCu<<<blocks, threads>>>(d_p0, beta_k, k);
		VecAddCu<<<blocks, threads>>>(d_r, d_p0, d_p0, k);
		residual0 =  residual2;
		iter++;
		cudaDeviceSynchronize();
	}

cleanup:
	// Finally, copy x_k1 to x and done

	if (!fail){
		cudaMemcpy(x, d_x, array_size, cudaMemcpyDeviceToHost);
	}
	

	// Do Memory Allocation cleaning

	// Host
	for (int i = 0; i < 1; i ++){
		free(host_pointers[i]);
		host_pointers[i] = NULL;
	}


	// device
	
	for (int i = 0; i < 5; i ++){
		cudaFree(device_pointers[i]);
		device_pointers[i] = NULL;
	}

	return iter;
}