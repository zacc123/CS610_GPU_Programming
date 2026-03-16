#ifndef UTILS_H
#define UTILS_H

// SPARSE Arrays

typedef struct {
    int rows;
    int cols;
    int nnz;
    int *row_indices;
    int *col_indices;
    double *values;
} SparseMatrixCOO;

/* Internal triplet used only for sorting */
typedef struct {
    int row;
    int col;
    double value;
} COOEntry;

float *create_1D_domain(float x0, float xN, int Nx);
void free_1D_domain(float **x);

double vectorDotProduct(double *a, double *b, int k);
void vectorSub(double *a, double *b, double *c, int k);
void vectorSubIP(double *a, double *b, int k);
void vectorAdd(double *a, double *b, double *c, int k);
void vectorAddIP(double *a, double *b, int k);
void vectorMulScalar(double *a, double b, double *c, int k);
void vectorMulScalarIP(double *a, double b, int k);
void vectorMulEW(double *a, double *b, double *c, int k);
double vectorSum(double *a, int k);

void matrixMulDense(double *a, double *b, double *c, int m, int n, int k);
void matrixAddDense(double *a, double *b, double *c, int m, int n);
void matrixSubDense(double *a, double *b, double *c, int m, int n);

void matrixAddDenseIP(double *a, double *b, int m, int n);

void matrixVecDense(double *A, double *x, double *b, int m, int n);

void matrixMulScalarIP(double *A, double c, int m, int n);
void matrixMulScalar(double *A, double *res, double c, int m, int n);
void matrixMulScalarIPSparse(SparseMatrixCOO *A, double c);

void matrixTranspose(double *A, double *res, int m, int n);
SparseMatrixCOO* matrixTransposeSparse(SparseMatrixCOO *A);

// int conjugateGradient(double *A, double *x, double *b, int k, double atol, int max_iter);

double clamp(double num, double tol);


SparseMatrixCOO* create_coo_matrix(int m, int n, int non_zeros);
void free_coo_matrix(SparseMatrixCOO *matrix);


int compare_coo_entries(const void *a, const void *b);
void sort_coo(SparseMatrixCOO *mat);
SparseMatrixCOO *liftOutSparse(SparseMatrixCOO *A, SparseMatrixCOO *B);
#endif //UTILS_H