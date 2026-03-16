#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../include/sbp.h"
#include "../include/utils.h"
#include "../include/matrix_free.cuh"


D2_mf *create_D2_mf(int bm, int bn, int interior, int bs, double hi, double *D, double *BS, double *BD){
    D2_mf *params = (D2_mf *)malloc(sizeof(D2_mf));

    if (params == NULL){
        fprintf(stderr, "Failed to Allocate D2mf\n");
        exit(EXIT_FAILURE);
    }
    

    params->bm = bm;
    params->bn = bn;
    params->bs = bs;
    params->interior = interior;
    params->hi = hi;

    params->D = D;
    params-> BS = BS;
    params->BD = BD;

    return params;
}

void freeD2_mf(D2_mf **d2_params){
    free(*d2_params);
    *d2_params = NULL;
}

/*
 *
 *
 * Crux of this is to not have to form A operator
 *  Need to do D2 action and SAT terms, we''ll start doing them separately
 */
void D2matrixFreeDir(double *x, double *result, int Nxp, D2_mf* d2_params){

    double hi = d2_params->hi;

    for (int idx = 0; idx < Nxp; idx++){
        result[idx] = 0.0; // Start by initializing result

        // Lets start with D2
        if (idx < d2_params->bm){
            for (int j = 0; j < d2_params->bn; j++){
                //printf("at index %u %u d2 = %.2f\n", i, j,  bd[i*bn + j] / h2);
                result[idx] += d2_params->BD[j + idx*d2_params->bn] * x[j];
            }
            if (idx == 0){
                result[idx] = -0.5 * hi * result[idx]; // Makes it SPD
            }
            else {
                result[idx] = -1.0 * hi * result[idx]; // Makes it SPD
            }
        }

        else if (idx > Nxp - d2_params->bm - 1){
            for (int j = 0; j < d2_params->bn; j++){
                result[idx] += d2_params->BD[(Nxp - idx - 1)*d2_params->bn + (d2_params->bn - j - 1)] * x[Nxp - j - 1];
            }
            if (idx == (Nxp-1)){
                result[idx] = -0.5 * hi * result[idx]; // Makes it SPD
            }
            else {
                result[idx] = -1.0 * hi * result[idx]; // Makes it SPD
            }
        }
 
        else{ // Apply interior stencil
            
            for (int j=0; j<d2_params->interior; j++){
                result[idx] +=  d2_params->D[j]*x[idx+ j - (d2_params->interior / 2)];
            }
            result[idx] = -1.0 * hi * result[idx]; // Makes it SPD

        }

    }
// Now add the DIR Term:

        double sat1 = -1.0* hi * (d2_params->BS[0] + (2 * -13.0));
        

        result[0] += sat1*x[0];
        result[Nxp-1] += sat1*x[Nxp-1];
        for (int i = 1; i<d2_params->bs; i++){

            sat1 = -1.0 *hi*(d2_params->BS[i]);
            //printf("SAT Terms: %f", sat1);
            result[i] += sat1*x[0];
            result[Nxp-i-1] += sat1* x[Nxp-1];
        }


}

/*
 *
 *
 * Crux of this is to not have to form A operator
 *  Need to do D2 action and SAT terms, we''ll start doing them separately
 */
__global__ void D2matrixFreeDirCu(double *x, double *result, int Nxp, double hi){

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

        int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

        if (tidx < Nxp){
            result[tidx] = 0.0; // Start by initializing result
            // Lets start with D2
            if (tidx < bm){
                for (int j = 0; j < bn; j++){
                    //printf("at index %u %u d2 = %.2f\n", i, j,  bd[i*bn + j] / h2);
                    result[tidx] += bd[j + tidx*bn] * x[j];
                }
                if (tidx == 0){
                    result[tidx] = -0.5 * hi * result[tidx]; // Makes it SPD
                }
                else {
                    result[tidx] = -1.0 * hi * result[tidx]; // Makes it SPD
                }
            }

            else if (tidx > Nxp - bm - 1){
                for (int j = 0; j < bn; j++){
                    result[tidx] += bd[(Nxp - tidx - 1)*bn + (bn - j - 1)] * x[Nxp - j - 1];
                }
                if (tidx == (Nxp-1)){
                    result[tidx] = -0.5 * hi * result[tidx]; // Makes it SPD
                }
                else {
                    result[tidx] = -1.0 * hi * result[tidx]; // Makes it SPD
                }
            }
        
            else{ // Apply interior stencil
                    
                for (int j=0; j<interior; j++){
                    result[tidx] +=  d[j]*x[tidx+ j - (interior / 2)];
                }
                result[tidx] = -1.0 * hi * result[tidx]; // Makes it SPD
            }

        }
        __syncthreads();
        if (tidx < Nxp){
            double sat1 = -1.0* hi * (bsd[0] + (2 * -13.0));
            if (tidx == 0 || tidx == (Nxp-1)){
                result[tidx] += sat1*x[tidx];
            }
            if (tidx == 0){
                for (int i = 1; i<bs; i++){
                    sat1 = -1.0 *hi*(bsd[i]);
                    //printf("SAT Terms: %f", sat1);
                    result[i] += sat1*x[0];
                    result[Nxp-i-1] += sat1* x[Nxp-1];
                }
            }
            else if (tidx == Nxp-1){
                for (int i = 1; i<bs; i++){
                    sat1 = -1.0 *hi*(bsd[i]);
                    //printf("SAT Terms: %f", sat1);
                    result[Nxp-i-1] += sat1* x[Nxp-1];
                }
            }

        }

}

void boundaryMF(double *b, D2_mf *d2_params, double *data, int Nxp){
    
    
    double hi = d2_params->hi;
    double sat1 = -1.0* hi * (d2_params->BS[0] + (2 * -13.0));
        

        b[0] += sat1*data[0];
        b[Nxp-1] += sat1*data[Nxp-1];
        for (int i = 1; i<d2_params->bs; i++){

            sat1 = -1.0 *hi*(d2_params->BS[i]);
            //printf("SAT Terms: %f", sat1);
            b[i] += sat1*data[0];
            b[Nxp-i-1] += sat1* data[Nxp-1];
        }
}

/*
 *
 *
 * Crux of this is to not have to form A operator
 *  Need to do D2 action and SAT terms, we''ll start doing them separately
 */
__global__ void D2matrixFreeDirCuOpt(double *x, double *result, int Nxp, double hi){

        double out = 0.0;

        int tidx = threadIdx.x + (blockIdx.x * blockDim.x);

        if (tidx < Nxp){
            result[tidx] = 0.0; // Start by initializing result
            // Lets start with D2
            if (tidx==0){
                out =  (x[0] - 2.0 * x[1] + x[2]) * 0.5;
                out += (1.5 + (2 * -13.0)) * x[0]; // SAT Term
            }

            else if (tidx == Nxp-1){
                out =  (x[Nxp-3] - 2.0 * x[Nxp-2] + x[Nxp-1]) * 0.5;
                out += (1.5 + (2 * -13.0)) * x[Nxp-1]; // SAT Term
            }
            else{ // Apply interior stencil
                out =  x[tidx - 1] - 2.0 * x[tidx] + x[tidx + 1];
            }

            if (tidx == 1){
                out += -2.0 * x[0]; 
            }
            else if (tidx == 2){
                out += 0.5 * x[0]; 
            }
            else if (tidx == Nxp-2){
                out += -2.0 * x[Nxp-1];
            }
            else if (tidx == Nxp-3){
                out += 0.5 * x[Nxp-1]; 
            }
        
            out *= -1.0*hi;
            result[tidx] = out;
        }

}

