
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/utils.h"
/* * * * * * * * * * * *
/
/ DOMAIN Helpers
/
/ * * * * * * * * * * */


/*
 * Function to create line vector of X points on a domain
 *      INPUT: Array pointer x, start and end (x0, xN), and number of points Nx
 *          
 */
float *create_1D_domain(float x0, float xN, unsigned int Nx){

    size_t size_x = (Nx + 1) * sizeof(float);
    float *x = (float *)malloc(size_x);

    if (x == NULL){
        fprintf(stderr, "Memory Allocation in Domain failed\n");
        exit(EXIT_FAILURE);
    }

    float dx = (xN - x0) / (float)Nx;
    for (unsigned int i = 0; i < Nx + 1; i++){
        (x)[i] = (float)i * dx;
    }

    return x;
}

// Clean Up wrapper for domain
void free_1D_domain(float **x){
    free(*x);
    *x = NULL;
}


/* * * * * * * * * * * *
/
/ Matrix Mul
/
/ * * * * * * * * * * */

/* CPU Version of Matrix Mul and validate */
void matrixMulDense(double *a, double *b, double *c, unsigned int m, unsigned int n, unsigned int k) {
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

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixAddDense(double *a, double *b, double *c, unsigned int m, unsigned int n) {
	int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			c[idx] = a[idx] + b[idx];
		}
		
	}
	// No return : )
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixAddDenseIP(double *a, double *b, unsigned int m, unsigned int n) {
	int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			a[idx] = a[idx] + b[idx];
		}
	}
	// No return : )
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixSubDense(double *a, double *b, double *c, unsigned int m, unsigned int n) {
	int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			c[idx] = a[idx] - b[idx];
		}
		
	}
	// No return : )
}

/* CPU Version of Matrix Mul and validate */
void matrixVecDense(double *A, double *x, double *b, unsigned int m, unsigned int n) {
	for (unsigned int i=0; i<m; i++){
        b[i] = 0.0;
		for (unsigned int j=0; j<n; j++){
			b[i] += A[j + i*n] * x[j];		 // Reset to 0
		}
	}
	// No return : )
}

void matrixMulScalar(double *A, double *res, double c, unsigned int m, unsigned int n){
    int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			res[idx] = A[idx]*c;
		}
		
	}
}

void matrixMulScalarIP(double *A, double c, unsigned int m, unsigned int n){
    int idx;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx = (i * m) + j;
			A[idx] = A[idx]*c;
		}
		
	}
}

void matrixTranspose(double *A, double *res, unsigned int m, unsigned int n){
    int idx_a, idx_b;
	for (unsigned int i=0; i<n; i++){
		for (unsigned int j=0; j<m; j++){
			idx_a = (i * m) + j;
            idx_b = i + (j * n);
			res[idx_b] = A[idx_a];
		}
		
	}
}
/* * * * * * * * * **** *** * * * ** * * * ** * **
 * 
 * Vector Functions returning scalars
 *
 * * * * * * * * * **** *** * * * ** * * * ** * **
 */



/*
 * Handle Dot product of vectors a'b where len of a == len b == k 
 */
double vectorDotProduct(double *a, double *b, unsigned int k){
	double sum = 0.0;

	for (unsigned int i = 0; i < k; i++){
		sum += a[i] * b[i];
	}
	return sum;
}
/* * * * * * * * * **** *** * * * ** * * * ** * **
 * 
 * Vector Functions returning void (not in-place)
 *
 * * * * * * * * * **** *** * * * ** * * * ** * **
 */

/*
 * Takes a - b and stores in vector c. All of length k
 */
void vectorSub(double *a, double *b, double *c, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		c[i] = a[i] - b[i];
	}
	
}

void vectorSubIP(double *a, double *b, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		a[i] = a[i] - b[i];
	}
	
}

/*
 * Takes a - b and stores in vector c. All of length k
 */
void vectorAdd(double *a, double *b, double *c, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		c[i] = a[i] + b[i];
	}
	
}

void vectorAddIP(double *a, double *b, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		a[i] = a[i] + b[i];
	}
	
}

void vectorMulScalar(double *a, double b, double *c, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		c[i] = a[i]*b;
	}
	
}

void vectorMulScalarIP(double *a, double b, unsigned int k){
	for (unsigned int i = 0; i < k; i++){
		a[i] = a[i]*b;
	}
	
}

int conjugateGradient(double *A, double *x, double *b, unsigned int k, double atol, unsigned int max_iter){
	
	// Setup needed vectors
	// Keep them together to not have to free each ind
	double *r_0, *r_1, *p_0, *p_1, *tmp_res, *tmp_res2, *x_k, *x_k1;
	double *pointers[8] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};
	double clamp_tol = 1e-16;
	double atol2 = atol * atol;


	size_t array_size = k * sizeof(double);
	for (unsigned int i = 0; i < 8; i ++){
		pointers[i] = (double *)malloc(array_size);
		if (pointers[i] == NULL){
			for (unsigned int j = 0; j < i; j++) free(pointers[j]);
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

	for (unsigned int i = 0; i < max_iter; i++){

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
	for (unsigned int i = 0; i < 8; i++){
		free(pointers[i]); pointers[i] = NULL;
	}
	return iter;
}

double clamp(double num, double tol){
	return (num > tol)? num : tol;
}
