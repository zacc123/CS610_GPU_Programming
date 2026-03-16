# Repo for CS610 GPU Programming Final Project
#### Author: Zac Cross
#### Date: 3/16/2026
---
## GPU Enabled Matrix-Free Kernels for 1D SBP-SAT Operators

**Contents:**

This repo contains the source code and result csv's from my project runs. Below gives a high level summary of the contents: 
1. CPU Versions of a Conjugate Gradient solver and necessary helper BLAS operations for dense matrix vector systems.

2. CUDA Enabled CG solver, compatible with dense matrix vector systems, and naive implementations of BLAS kernels.

3. CUDA kernel optmizations for necessary BLAS operations, utilizing tiling with shared memory and register use reduction.

4. SBP-SAT code for a 1D wave equation simulation, with a MMS and convergence testing for orders p=2,4,and 6 for a dense operator.

5. A matrix free SBP SAT operator for the CPU and GPU CG methods for order p=2. 

6. Testing for wall clock time to execute for all of the above operations.

---
**How to Run**

Unzip all files and run command "make clean all" . The makefile should handle the rest targeting a Compute_80 NVIDIA GPU architecture.
---

**Index for relevant files in the source code**
Below gives a brief explanation of what is in each of the files in src. The rest of the documentation can be found in the respective files.

1. *main.cu*: Holds the main program, calling into the different testing functions for timing. **Grid sizes for testing can be seen here**.

2. *convergence.cu*: Holds the framework used for convergence tests for dense and matrix-free operators. 

3. *utils.cpp*: Holds CPU versions of Linear Algebra functions, grid + domain helper functions, and all other misc helpers. 

4. *sbp.cpp*: Contains SBP and SAT Operator creation for Dense operators.

5. *cg.cu*: Contains naive and optimized CG solvers for CPU, GPU, and GPU Matrix-Free.

6. *matrix_free.cu*: Holds CPU and GPU Matrix-Free function/kernels.

7. *matrix_ops.cu*: Contains GPU kerenels for Linear Algebra operations.

8. *tests.cu*: Frameworks for all testing of correctness and speed. Tests Linear algebra operations and CG solvers, with CPU for validation, for correctness and time to execute (wall clock time)

9. *timing.cpp*: Contains helpers for timing CPU calls from Prof Choi's CS531 class.