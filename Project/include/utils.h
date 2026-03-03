#ifndef UTILS_H
#define UTILS_H

// SPARSE Arrays

typedef struct {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int *row_indices;
    unsigned int *col_indices;
    double *values;
} SparseMatrixCOO;

/* Internal triplet used only for sorting */
typedef struct {
    unsigned int row;
    unsigned int col;
    double value;
} COOEntry;

float *create_1D_domain(float x0, float xN, unsigned int Nx);
void free_1D_domain(float **x);

double vectorDotProduct(double *a, double *b, unsigned int k);
void vectorSub(double *a, double *b, double *c, unsigned int k);
void vectorSubIP(double *a, double *b, unsigned int k);
void vectorAdd(double *a, double *b, double *c, unsigned int k);
void vectorAddIP(double *a, double *b, unsigned int k);
void vectorMulScalar(double *a, double b, double *c, unsigned int k);
void vectorMulScalarIP(double *a, double b, unsigned int k);

void matrixMulDense(double *a, double *b, double *c, unsigned int m, unsigned int n, unsigned int k);
void matrixAddDense(double *a, double *b, double *c, unsigned int m, unsigned int n);
void matrixSubDense(double *a, double *b, double *c, unsigned int m, unsigned int n);

void matrixAddDenseIP(double *a, double *b, unsigned int m, unsigned int n);

void matrixVecDense(double *A, double *x, double *b, unsigned int m, unsigned int n);

void matrixMulScalarIP(double *A, double c, unsigned int m, unsigned int n);
void matrixMulScalar(double *A, double *res, double c, unsigned int m, unsigned int n);
void matrixMulScalarIPSparse(SparseMatrixCOO *A, double c);

void matrixTranspose(double *A, double *res, unsigned int m, unsigned int n);
SparseMatrixCOO* matrixTransposeSparse(SparseMatrixCOO *A);

int conjugateGradient(double *A, double *x, double *b, unsigned int k, double atol, unsigned int max_iter);

double clamp(double num, double tol);


SparseMatrixCOO* create_coo_matrix(unsigned int m, unsigned int n, unsigned int non_zeros);
void free_coo_matrix(SparseMatrixCOO *matrix);


int compare_coo_entries(const void *a, const void *b);
void sort_coo(SparseMatrixCOO *mat);
SparseMatrixCOO *liftOutSparse(SparseMatrixCOO *A, SparseMatrixCOO *B);
#endif //UTILS_H