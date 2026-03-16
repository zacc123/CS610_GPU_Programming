
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


/*
 *
 *
 * SPARSE ARRAYs
 *
 *
 */

 SparseMatrixCOO* create_coo_matrix(int m, int n, int non_zeros) {
    SparseMatrixCOO *matrix = (SparseMatrixCOO*) malloc(sizeof(SparseMatrixCOO));
    if (matrix == NULL) return NULL;

    matrix->rows = m;
    matrix->cols = n;
    matrix->nnz = non_zeros;
    matrix->row_indices = (int*) malloc(non_zeros * sizeof(int));
    matrix->col_indices = (int*) malloc(non_zeros * sizeof(int));
    matrix->values = (double*) malloc(non_zeros * sizeof(double));

    // Error checking for malloc omitted for brevity
    if (matrix->row_indices == NULL || matrix->col_indices == NULL || matrix->values == NULL){
		fprintf(stderr, "Memory Allocation in Sparse Array\n");
        exit(EXIT_FAILURE);
	}
    return matrix;
}


// Don't forget to free the allocated memory when done
void free_coo_matrix(SparseMatrixCOO *matrix) {
    free(matrix->row_indices);
    free(matrix->col_indices);
    free(matrix->values);
    free(matrix);
}

/* Comparator: row-major then col */
int compare_coo_entries(const void *a, const void *b)
{
    const COOEntry *ea = (const COOEntry*)a;
    const COOEntry *eb = (const COOEntry*)b;

    if (ea->row < eb->row) return -1;
    if (ea->row > eb->row) return  1;

    if (ea->col < eb->col) return -1;
    if (ea->col > eb->col) return  1;

    return 0;
}

/* Sort COO in-place by row then column */
void sort_coo(SparseMatrixCOO *mat)
{
    if (!mat || mat->nnz == 0) return;

    int nnz = mat->nnz;

    /* Pack into temporary array */
    COOEntry *entries = (COOEntry*)malloc(nnz * sizeof(COOEntry));
    if (!entries) return;

    for (int i = 0; i < nnz; ++i) {
        entries[i].row   = mat->row_indices[i];
        entries[i].col   = mat->col_indices[i];
        entries[i].value = mat->values[i];
    }

    /* Sort */
    qsort(entries, nnz, sizeof(COOEntry), compare_coo_entries);

    /* Unpack back */
    for (int i = 0; i < nnz; ++i) {
        mat->row_indices[i] = entries[i].row;
        mat->col_indices[i] = entries[i].col;
        mat->values[i]      = entries[i].value;
    }

    free(entries);
}

/*
 * Use this for E0 and En
 *
 */
SparseMatrixCOO *liftOutSparse(SparseMatrixCOO *A, SparseMatrixCOO *B){
	int row = B->row_indices[0];

	// 2 pass since this will be at most 1 col of A
	int nnz = 0;
	double *nnzs = (double *)malloc(A->rows *sizeof(double));
	int *rows = (int *)malloc(A->rows *sizeof(int));
	int *cols = (int *)malloc(A->rows *sizeof(int));
	

	for (int i = 0; i < A->nnz; i++){
		if (A->col_indices[i] == row){
			nnzs[nnz] = A->values[i];
			rows[nnz] = A->row_indices[i];
			cols[nnz] = row;
			nnz++;
		}
	}

	SparseMatrixCOO *C = create_coo_matrix(A->rows, B->cols, nnz+1);
	memcpy(C->row_indices, rows, (nnz+1)*sizeof(int));
	memcpy(C->col_indices, cols, (nnz+1)*sizeof(int));
	memcpy(C->values, nnzs, (nnz+1)*sizeof(double));

	sort_coo(C);

	free(nnzs); free(rows); free(cols);

	return C;

}

void matrixMulScalarIPSparse(SparseMatrixCOO *A, double c){
	for (int i = 0; i < A->nnz; i++){
		A->values[i] = A->values[i] * c;
	}
}

SparseMatrixCOO* matrixTransposeSparse(SparseMatrixCOO *A){
	SparseMatrixCOO *AT = create_coo_matrix(A->cols, A->rows, A->nnz);
	memcpy(AT->row_indices, A->col_indices, A->nnz*sizeof(int));
	memcpy(AT->col_indices, A->row_indices, A->nnz*sizeof(int));
	memcpy(AT->values, A->values, A->nnz*sizeof(double));
	sort_coo(AT);
	return AT;
}

// SparseMatrixCOO* matrixMulSparse(SparseMatrixCOO *A, SparseMatrixCOO *B){
// 	SparseMatrixCOO *BT = matrixTransposeSparse(B);

// 	int c_row = 0;
// 	int c_col = 0;
// 	double c_val = 0.0;
// 	int c_vals;
	
// 	int res_cap = A->nnz;

// 	double *c_nnzs = (double *)malloc(sizeof(double) * res_cap);
// 	int *c_rows = (int *)malloc(sizeof(int) * res_cap);
// 	int *c_cols = (int *)malloc(sizeof(int) * res_cap);
	
// 	int A_row = A->row_indices[0];
// 	int B_col = B->row_indices[0];

// 	int a_idx = 0;
// 	int b_idx = 0;

// 	int a_idx_prev = 0;
// 	int b_idx_prev = 0;

// 	int c_idx = 0;

// 	for (int i = 0; i < A->row; i++){
		
// 		for (int j = 0; j < B->col; j++){
			
// 			// Do the sparse dot product between a at row i and bt and row i

// 			// START HERE
// 			if (A_row == i){
				
// 				b_idx_prev = b_idx;
// 				while (BT->row_indices[b_idx] == j)


// 			}



// 		}
// 	}

	


// 	return C;
// }
