
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

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
float *create_1D_domain(float x0, float xN, int Nx){

    size_t size_x = (Nx + 1) * sizeof(float);
    float *x = (float *)malloc(size_x);

    if (x == NULL){
        fprintf(stderr, "Memory Allocation in Domain failed\n");
        exit(EXIT_FAILURE);
    }

    float dx = (xN - x0) / (float)Nx;
    for (int i = 0; i < Nx + 1; i++){
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
void matrixMulDense(double *a, double *b, double *c, int m, int n, int k) {
	int idx;
	int a_idx, b_idx;

	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			
			idx = j + (i)*n; // Calc Idx of Array
			c[idx] = 0;		 // Reset to 0
			for (int _k=0; _k<k; _k++){
				a_idx = (i * k) + _k;
				b_idx = (_k * n) + j;
				c[idx] += a[a_idx] * b[b_idx];
			}
		}
	}
	// No return : )
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixAddDense(double *a, double *b, double *c, int m, int n) {
	int idx;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			idx = (i * m) + j;
			c[idx] = a[idx] + b[idx];
		}
		
	}
	// No return : )
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixAddDenseIP(double *a, double *b, int m, int n) {
	int idx;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			idx = (i * m) + j;
			a[idx] = a[idx] + b[idx];
		}
	}
	// No return : )
}

/* Exercise 2.1: CPU Version of Vector Add and validate */
void matrixSubDense(double *a, double *b, double *c, int m, int n) {
	int idx;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			idx = (i * m) + j;
			c[idx] = a[idx] - b[idx];
		}
		
	}
	// No return : )
}

/* CPU Version of Matrix Mul and validate */
void matrixVecDense(double *A, double *x, double *b, int m, int n) {
	for (int i=0; i<m; i++){
        b[i] = 0.0;
		for (int j=0; j<n; j++){
			b[i] += A[j + i*n] * x[j];		 // Reset to 0
		}
	}
	// No return : )
}

void matrixMulScalar(double *A, double *res, double c, int m, int n){
    int idx;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			idx = (i * m) + j;
			res[idx] = A[idx]*c;
		}
		
	}
}

void matrixMulScalarIP(double *A, double c, int m, int n){
    int idx;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
			idx = (i * m) + j;
			A[idx] = A[idx]*c;
		}
		
	}
}

void matrixTranspose(double *A, double *res, int m, int n){
    int idx_a, idx_b;
	for (int i=0; i<n; i++){
		for (int j=0; j<m; j++){
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
double vectorDotProduct(double *a, double *b, int k){
	double sum = 0.0;

	for (int i = 0; i < k; i++){
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
void vectorSub(double *a, double *b, double *c, int k){
	for (int i = 0; i < k; i++){
		c[i] = a[i] - b[i];
	}
	
}

void vectorSubIP(double *a, double *b, int k){
	for (int i = 0; i < k; i++){
		a[i] = a[i] - b[i];
	}
}

/*
 * Takes a - b and stores in vector c. All of length k
 */
void vectorAdd(double *a, double *b, double *c, int k){
	for (int i = 0; i < k; i++){
		c[i] = a[i] + b[i];
	}
	
}

void vectorAddIP(double *a, double *b, int k){
	for (int i = 0; i < k; i++){
		a[i] = a[i] + b[i];
	}
	
}

void vectorMulScalar(double *a, double b, double *c, int k){
	for (int i = 0; i < k; i++){
		c[i] = a[i]*b;
	}
	
}

void vectorMulScalarIP(double *a, double b, int k){
	for (int i = 0; i < k; i++){
		a[i] = a[i]*b;
	}
	
}

double vectorSum(double *a, int k){
	double res = 0.0;
	for (int i = 0; i < k; i++){
		res += a[i];
	}
	return res;
}

void vectorMulEW(double *a, double *b, double *c, int k){
	for (int i = 0; i < k; i++){
		c[i] = a[i]*b[i];
	}
}

double clamp(double num, double tol){
	return (num > tol)? num : tol;
}
