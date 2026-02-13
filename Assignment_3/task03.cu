#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>


#define IMAGE_DIM 2048
#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8

#define BLOCK_SIZE 8
#define R1 1
#define R2 2
#define R4 4
#define R8 8

void output_image_file(uchar3* image, const char *filename);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);
void validate_image_file(uchar3* ref_image, uchar3* image);
void cpu_image_blur_A(uchar3 *image, uchar3 *image_output);
void valid_CpuGpu(uchar3 *ref_image, uchar3 *test_image);
void cpu_image_blur(uchar3 *image, uchar3 *image_output, int r);

using uchar = unsigned char;

__global__ void image_blur_A(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    int tid = tidx + tidy * IMAGE_DIM;
    int result_r = 0; // store res in register
    int result_g = 0;
    int result_b = 0;


    int x, y;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){    
        // Plus or minus x
        for (int i = -1; i<2; i++){
            
            y = tidy + i;

            // Cover Boundaries
            if (y < 0)
                y+= IMAGE_DIM;
            if (y >= IMAGE_DIM)
                y-= IMAGE_DIM;

            for (int j = -1; j<2; j++){

                // Cover X boundaries
                x = tidx + j;
                if (x < 0)
                    x+= IMAGE_DIM;
                if (x >= IMAGE_DIM)
                    x-= IMAGE_DIM;

                uchar3 pixel = image[x + y*IMAGE_DIM];
                result_r += pixel.x;
                result_g += pixel.y;
                result_b += pixel.z;
            }

        }
        result_r =  result_r / 9 ;
        result_g = result_g / 9 ;
        result_b = result_b / 9;

        //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
        image_output[tid] = make_uchar3(result_r, result_g, result_b);



    }
}

__global__ void image_blur_A_tiled(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
    int x, y;
    int r= 1;

    int global_x0 = blockIdx.x * blockDim.x;
    int global_y0 = blockIdx.y * blockDim.y;
    int tile_width = BLOCK_SIZE + 2*R1;
    // Shared memory stuff
    __shared__ uchar3 img_shared[(BLOCK_SIZE + R1 ) * (BLOCK_SIZE + R1)];
    // Assume each thread will work on its adjust mem location with row = [halo r img img img r halo]
    // Idea: rastor block over in x then down in y and over until all data is loaded
    // Use these to track how many times we've shifted
    for (int local_y = threadIdx.y; local_y < tile_width; local_y += blockDim.y) {

        int global_y = local_y + global_y0 - r;

        if (global_y < 0){
            global_y += IMAGE_DIM;
        }
        if (global_y >= IMAGE_DIM){
            global_y -= IMAGE_DIM;
        }

        for (int local_x = threadIdx.x; local_x < tile_width; local_x += blockDim.x) {

            int global_x = local_x + global_x0 - r;

            if (global_x < 0){
                global_x += IMAGE_DIM;
            }
            if (global_x >= IMAGE_DIM){
                global_x -= IMAGE_DIM;
            }

            img_shared[local_x + local_y*tile_width] = image[global_x + global_y*IMAGE_DIM];

        }


    }

    __syncthreads(); // Now everyone has loaded in their data

    int tidx = global_x0 + threadIdx.x; 
    int tidy = global_y0 + threadIdx.y; 

    if (tidy >= IMAGE_DIM || tidx >= IMAGE_DIM){
        return;
    }

    // Plus or minus x
    int result_r = 0;
    int result_b = 0;
    int result_g = 0;

    for (int i = 0-r; i<1+r; i++){
            
        y = threadIdx.y + i + r;

        for (int j = 0-r; j<1+r; j++){

            // Cover X boundaries
            x = threadIdx.x + j + r;

            uchar3 pixel = img_shared[x + y*tile_width];
            result_r += pixel.x;
            result_g += pixel.y;
            result_b += pixel.z;
        }

    }
    result_r =  result_r / 9 ;
    result_g = result_g / 9 ;
    result_b = result_b / 9;

    //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
    image_output[tidx + IMAGE_DIM*tidy] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);

    
}

__global__ void image_blur_B(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	 int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	 int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
     int tid = tidx + tidy * IMAGE_DIM;
    int result_r = 0; // store res in register
    int result_g = 0;
    int result_b = 0;

    int x, y;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){    
        // Plus or minus x
        for (int i = -2; i<3; i++){
            
            y = tidy + i;

            // Cover Boundaries
            if (y < 0)
                y+= IMAGE_DIM;
            if (y >= IMAGE_DIM)
                y-= IMAGE_DIM;

            for (int j = -2; j<3; j++){

                // Cover X boundaries
                x = tidx + j;
                if (x < 0)
                    x+= IMAGE_DIM;
                if (x >= IMAGE_DIM)
                    x-= IMAGE_DIM;

                uchar3 pixel = image[x + y*IMAGE_DIM];
                result_r += pixel.x;
                result_g += pixel.y;
                result_b += pixel.z;
            }

        }
        result_r =  result_r / 25 ;
        result_g = result_g / 25 ;
        result_b = result_b / 25;

        //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
        image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);

    }
	
}

__global__ void image_blur_B_tiled(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
    int x, y;
    int r= 2;

    int global_x0 = blockIdx.x * blockDim.x;
    int global_y0 = blockIdx.y * blockDim.y;
    int tile_width = BLOCK_SIZE + 2*R2;
    // Shared memory stuff
    __shared__ uchar3 img_shared[(BLOCK_SIZE + R2 ) * (BLOCK_SIZE + R2)];
    // Assume each thread will work on its adjust mem location with row = [halo r img img img r halo]
    // Idea: rastor block over in x then down in y and over until all data is loaded
    // Use these to track how many times we've shifted
    for (int local_y = threadIdx.y; local_y < tile_width; local_y += blockDim.y) {

        int global_y = local_y + global_y0 - r;

        if (global_y < 0){
            global_y += IMAGE_DIM;
        }
        if (global_y >= IMAGE_DIM){
            global_y -= IMAGE_DIM;
        }

        for (int local_x = threadIdx.x; local_x < tile_width; local_x += blockDim.x) {

            int global_x = local_x + global_x0 - r;

            if (global_x < 0){
                global_x += IMAGE_DIM;
            }
            if (global_x >= IMAGE_DIM){
                global_x -= IMAGE_DIM;
            }

            img_shared[local_x + local_y*tile_width] = image[global_x + global_y*IMAGE_DIM];

        }


    }

    __syncthreads(); // Now everyone has loaded in their data

    int tidx = global_x0 + threadIdx.x; 
    int tidy = global_y0 + threadIdx.y; 

    if (tidy >= IMAGE_DIM || tidx >= IMAGE_DIM){
        return;
    }

    // Plus or minus x
    int result_r = 0;
    int result_b = 0;
    int result_g = 0;

    for (int i = 0-r; i<1+r; i++){
            
        y = threadIdx.y + i + r;

        for (int j = 0-r; j<1+r; j++){

            // Cover X boundaries
            x = threadIdx.x + j + r;

            uchar3 pixel = img_shared[x + y*tile_width];
            result_r += pixel.x;
            result_g += pixel.y;
            result_b += pixel.z;
        }

    }
    result_r =  result_r / ((2*r + 1) * (2*r + 1)) ;
    result_g = result_g / ((2*r + 1) * (2*r + 1)) ;
    result_b = result_b / ((2*r + 1) * (2*r + 1));

    //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
    image_output[tidx + IMAGE_DIM*tidy] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);

}

__global__ void image_blur_C(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	 int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	 int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
     int tid = tidx + tidy * IMAGE_DIM;
    int result_r = 0; // store res in register
    int result_g = 0;
    int result_b = 0;
    int r = 4;

    int x, y;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){    
        // Plus or minus x
        for (int i = 0-r; i<r+1; i++){
            
            y = tidy + i;

            // Cover Boundaries
            if (y < 0)
                y+= IMAGE_DIM;
            if (y >= IMAGE_DIM)
                y-= IMAGE_DIM;

            for (int j = 0 - r; j<1 + r; j++){

                // Cover X boundaries
                x = tidx + j;
                if (x < 0)
                    x+= IMAGE_DIM;
                if (x >= IMAGE_DIM)
                    x-= IMAGE_DIM;

                uchar3 pixel = image[x + y*IMAGE_DIM];
                result_r += pixel.x;
                result_g += pixel.y;
                result_b += pixel.z;
            }

        }
        result_r =  result_r / ((2*r + 1) * (2*r + 1)) ;
        result_g = result_g / ((2*r + 1) * (2*r + 1)) ;
        result_b = result_b / ((2*r + 1) * (2*r + 1));

        //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
        image_output[tid] = make_uchar3(result_r, result_g, result_b);



    }
	
}

__global__ void image_blur_C_tiled(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
    int x, y;
    int r= 4;

    int global_x0 = blockIdx.x * blockDim.x;
    int global_y0 = blockIdx.y * blockDim.y;
    int tile_width = BLOCK_SIZE + 2*r;
    // Shared memory stuff
    __shared__ uchar3 img_shared[(BLOCK_SIZE + R4) * (BLOCK_SIZE + R4)];
    // Assume each thread will work on its adjust mem location with row = [halo r img img img r halo]
    // Idea: rastor block over in x then down in y and over until all data is loaded
    // Use these to track how many times we've shifted
    for (int local_y = threadIdx.y; local_y < tile_width; local_y += blockDim.y) {

        int global_y = local_y + global_y0 - r;

        if (global_y < 0){
            global_y += IMAGE_DIM;
        }
        if (global_y >= IMAGE_DIM){
            global_y -= IMAGE_DIM;
        }

        for (int local_x = threadIdx.x; local_x < tile_width; local_x += blockDim.x) {

            int global_x = local_x + global_x0 - r;

            if (global_x < 0){
                global_x += IMAGE_DIM;
            }
            if (global_x >= IMAGE_DIM){
                global_x -= IMAGE_DIM;
            }

            img_shared[local_x + local_y*tile_width] = image[global_x + global_y*IMAGE_DIM];

        }


    }

    __syncthreads(); // Now everyone has loaded in their data

    int tidx = global_x0 + threadIdx.x; 
    int tidy = global_y0 + threadIdx.y; 

    if (tidy >= IMAGE_DIM || tidx >= IMAGE_DIM){
        return;
    }

    // Plus or minus x
    int result_r = 0;
    int result_b = 0;
    int result_g = 0;

    for (int i = 0-r; i<1+r; i++){
            
        y = threadIdx.y + i + r;

        for (int j = 0-r; j<1+r; j++){

            // Cover X boundaries
            x = threadIdx.x + j + r;

            uchar3 pixel = img_shared[x + y*tile_width];
            result_r += pixel.x;
            result_g += pixel.y;
            result_b += pixel.z;
        }

    }
    result_r =  result_r / ((2*r + 1) * (2*r + 1)) ;
    result_g = result_g / ((2*r + 1) * (2*r + 1)) ;
    result_b = result_b / ((2*r + 1) * (2*r + 1));

    //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
    image_output[tidx + IMAGE_DIM*tidy] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);

}

__global__ void image_blur_D(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
	// Add your implementation here
	 int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	 int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
     int tid = tidx + tidy * IMAGE_DIM;
    int result_r = 0; // store res in register
    int result_g = 0;
    int result_b = 0;
    int r = 8;

    int x, y;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){    
        // Plus or minus x
        for (int i = 0-r; i<r+1; i++){
            
            y = tidy + i;

            // Cover Boundaries
            if (y < 0)
                y+= IMAGE_DIM;
            if (y >= IMAGE_DIM)
                y-= IMAGE_DIM;

            for (int j = 0 - r; j<1 + r; j++){

                // Cover X boundaries
                x = tidx + j;
                if (x < 0)
                    x+= IMAGE_DIM;
                if (x >= IMAGE_DIM)
                    x-= IMAGE_DIM;

                uchar3 pixel = image[x + y*IMAGE_DIM];
                result_r += pixel.x;
                result_g += pixel.y;
                result_b += pixel.z;
            }

        }
        result_r =  result_r / ((2*r + 1) * (2*r + 1)) ;
        result_g = result_g / ((2*r + 1) * (2*r + 1)) ;
        result_b = result_b / ((2*r + 1) * (2*r + 1));

        //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
        image_output[tid] = make_uchar3(result_r, result_g, result_b);



    }
}

__global__ void image_blur_D_tiled(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
    int x, y;
    int r= 8;

    int global_x0 = blockIdx.x * blockDim.x;
    int global_y0 = blockIdx.y * blockDim.y;
    int tile_width = BLOCK_SIZE + 2*r;
    // Shared memory stuff
    __shared__ uchar3 img_shared[(BLOCK_SIZE + R8) * (BLOCK_SIZE + R8)];
    // Assume each thread will work on its adjust mem location with row = [halo r img img img r halo]
    // Idea: rastor block over in x then down in y and over until all data is loaded
    // Use these to track how many times we've shifted
    for (int local_y = threadIdx.y; local_y < tile_width; local_y += blockDim.y) {

        int global_y = local_y + global_y0 - r;

        if (global_y < 0){
            global_y += IMAGE_DIM;
        }
        if (global_y >= IMAGE_DIM){
            global_y -= IMAGE_DIM;
        }

        for (int local_x = threadIdx.x; local_x < tile_width; local_x += blockDim.x) {

            int global_x = local_x + global_x0 - r;

            if (global_x < 0){
                global_x += IMAGE_DIM;
            }
            if (global_x >= IMAGE_DIM){
                global_x -= IMAGE_DIM;
            }

            img_shared[local_x + local_y*tile_width] = image[global_x + global_y*IMAGE_DIM];

        }


    }

    __syncthreads(); // Now everyone has loaded in their data

    int tidx = global_x0 + threadIdx.x; 
    int tidy = global_y0 + threadIdx.y; 

    if (tidy >= IMAGE_DIM || tidx >= IMAGE_DIM){
        return;
    }

    // Plus or minus x
    int result_r = 0;
    int result_b = 0;
    int result_g = 0;

    for (int i = 0-r; i<1+r; i++){
            
        y = threadIdx.y + i + r;

        for (int j = 0-r; j<1+r; j++){

            // Cover X boundaries
            x = threadIdx.x + j + r;

            uchar3 pixel = img_shared[x + y*tile_width];
            result_r += pixel.x;
            result_g += pixel.y;
            result_b += pixel.z;
        }

    }
    result_r =  result_r / ((2*r + 1) * (2*r + 1)) ;
    result_g = result_g / ((2*r + 1) * (2*r + 1)) ;
    result_b = result_b / ((2*r + 1) * (2*r + 1));

    //image_output[tid] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);
    image_output[tidx + IMAGE_DIM*tidy] = make_uchar3((uchar)result_r, (uchar)result_g, (uchar)result_b);

}


/* Host code */

int main(void) {
	unsigned int image_size;
	uchar3 *d_image, *d_image_output;
	uchar3 *h_image, *h_image_output, *h_image_ref;
	cudaEvent_t startA, stopA, startB, stopB, startC, stopC, startD, stopD;
    cudaEvent_t startAt, stopAt, startBt, stopBt, startCt, stopCt, startDt, stopDt;
	float msA, msB, msC, msD;
    float msAt, msBt, msCt, msDt;

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar3);

	// create timers
    cudaEventCreate(&startA); cudaEventCreate(&stopA);
    cudaEventCreate(&startB); cudaEventCreate(&stopB);
    cudaEventCreate(&startC); cudaEventCreate(&stopC);
    cudaEventCreate(&startD); cudaEventCreate(&stopD);

    cudaEventCreate(&startAt); cudaEventCreate(&stopAt);
    cudaEventCreate(&startBt); cudaEventCreate(&stopBt);
    cudaEventCreate(&startCt); cudaEventCreate(&stopCt);
    cudaEventCreate(&startDt); cudaEventCreate(&stopDt);
    

	// allocate memory on the GPU for the output image
    cudaMalloc((void **)&d_image, image_size);
    cudaMalloc((void **)&d_image_output, image_size);
	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar3*)malloc(image_size);
    h_image_output = (uchar3*)malloc(image_size);
    h_image_ref = (uchar3*)malloc(image_size);
	input_image_file("input.ppm", h_image);

	// copy image to device memory
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	// launch kernel
    // launch kernel
    // Setup Kernel Launch Params
	int numBlocksX = (IMAGE_DIM + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
	int numBlocksY = (IMAGE_DIM + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;

	dim3 numBlocks(numBlocksX, numBlocksY, 1);
	dim3 numThreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	/************************************************************/
    /* Running All 8 kernels here in 1 go: */
	// warm up A:
	image_blur_A<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startA, 0); // Using the default stream (stream 0)
    image_blur_A<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopA, 0);
    cudaEventSynchronize(stopA);
    cudaEventElapsedTime(&msA, startA, stopA);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 1);
    printf("Checking A Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *a1 = "task03_A.ppm";
	output_image_file(h_image_output, a1);

    // warm up A Tiled:
	image_blur_A_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startAt, 0); // Using the default stream (stream 0)
    image_blur_A_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopAt, 0);
    cudaEventSynchronize(stopAt);
    cudaEventElapsedTime(&msAt, startAt, stopAt);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 1);
    printf("Checking Tiled A Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *a2 = "task03_A_tiled.ppm";
	output_image_file(h_image_output, a2);

    /* *********** */
    /* B */
    // warm up A:
	image_blur_B<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startB, 0); // Using the default stream (stream 0)
    image_blur_B<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopB, 0);
    cudaEventSynchronize(stopB);
    cudaEventElapsedTime(&msB, startB, stopB);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 2);
    printf("Checking B Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *b1 = "task03_B.ppm";
	output_image_file(h_image_output, b1);

    // warm up A Tiled:
	image_blur_B_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startBt, 0); // Using the default stream (stream 0)
    image_blur_B_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopBt, 0);
    cudaEventSynchronize(stopBt);
    cudaEventElapsedTime(&msBt, startBt, stopBt);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 2);
    printf("Checking Tiled B Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *b2 = "task03_B_tiled.ppm";
	output_image_file(h_image_output, b2);

    /* *********** */
    /* C */
    // warm up C:
	image_blur_C<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startC, 0); // Using the default stream (stream 0)
    image_blur_C<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopC, 0);
    cudaEventSynchronize(stopC);
    cudaEventElapsedTime(&msC, startC, stopC);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 4);
    printf("Checking C Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *c1 = "task03_C.ppm";
	output_image_file(h_image_output, c1);

    // warm up C Tiled:
	image_blur_C_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startCt, 0); // Using the default stream (stream 0)
    image_blur_C_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopCt, 0);
    cudaEventSynchronize(stopCt);
    cudaEventElapsedTime(&msCt, startCt, stopCt);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 4);
    printf("Checking Tiled C Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *c2 = "task03_C_tiled.ppm";
	output_image_file(h_image_output, c2);
    
    /* *********** */
    /* D */
    // warm up D:
	image_blur_D<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startD, 0); // Using the default stream (stream 0)
    image_blur_D<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopD, 0);
    cudaEventSynchronize(stopD);
    cudaEventElapsedTime(&msD, startD, stopD);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 8);
    printf("Checking D Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *d1 = "task03_D.ppm";
	output_image_file(h_image_output, d1);

    // warm up C Tiled:
	image_blur_D_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();
	checkCUDAError("Kernel launch");

    cudaEventRecord(startDt, 0); // Using the default stream (stream 0)
    image_blur_D_tiled<<<numBlocks, numThreads>>>(d_image, d_image_output);
    cudaEventRecord(stopDt, 0);
    cudaEventSynchronize(stopDt);
    cudaEventElapsedTime(&msDt, startDt, stopDt);
	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	// Validate with CPU
    cpu_image_blur(h_image, h_image_ref, 8);
    printf("Checking Tiled D Blur Against CPU....");
    valid_CpuGpu(h_image_ref, h_image_output);

    // Output Image
    const char *d2 = "task03_D_tiled.ppm";
	output_image_file(h_image_output, d2);

    // Output Image
    const char *d3 = "task03_D_cpu.ppm";
	output_image_file(h_image_ref, d3);
    
    
    //output timings
	printf("Execution time (ms):\n");
	printf("A:      \t%f\n", msA);
    printf("A-tiled:\t%f\n", msAt);
    printf("B:      \t%f\n", msB);
    printf("B-tiled:\t%f\n", msBt);
    printf("C:      \t%f\n", msC);
    printf("C-tiled:\t%f\n", msCt);
    printf("D:      \t%f\n", msD);
    printf("D-tiled:\t%f\n", msDt);
	
	
    //cleanup
    cudaEventDestroy(startA); cudaEventDestroy(stopA);
    
	
    cudaFree(d_image); cudaFree(d_image_output);
    free(h_image); free(h_image_output); free(h_image_ref);

	return 0;
}

void output_image_file(uchar3* image, const char *filename)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	f = fopen(filename, "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P6\n");
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int y = 0; y < IMAGE_DIM; y++){
		for (int x = 0; x < IMAGE_DIM; x++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
}

void input_image_file(char* filename, uchar3* image)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int x, y, s;

	//open the input file and write header info for PPM filetype
	f = fopen("input.ppm", "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'input.ppm' input file\n");
		exit(1);
	}
	fscanf(f, "%s\n", temp);
	fscanf(f, "%d\n%d\n", &x, &y);
	fscanf(f, "%d\n",&s);

	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		exit(1);
	}

	for (int y = 0; y < IMAGE_DIM; y++){
		for (int x = 0; x < IMAGE_DIM; x++){
			int i = x + y*IMAGE_DIM;
			fread(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void validate_image_file(uchar3* ref_image, uchar3* image)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int x, y, s;

	//open the output file and write header info for PPM filetype
	f = fopen("t3_r1_output.ppm", "rb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'input.ppm' input file\n");
		exit(1);
	}
	fscanf(f, "%s\n", temp);
	fscanf(f, "%d %d\n", &x, &y);
	fscanf(f, "%d\n",&s);
	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error: Input image file has wrong fixed dimensions\n");
		fprintf(stderr, "%d, %d\n", x, y);
		exit(1);
	}

	for (int y = 0; y < IMAGE_DIM; y++){
		for (int x = 0; x < IMAGE_DIM; x++){
			int i = x + y*IMAGE_DIM;
			fread(&image[i], sizeof(unsigned char), 3, f);
		}
	}

	fclose(f);

	FILE *ref_f; //input file handle
	//open the output file and write header info for PPM filetype
	ref_f = fopen("task03_correct_output_radius_1.ppm", "rb");
	if (ref_f == NULL){
		fprintf(stderr, "Error opening 'task2_corrected_output.ppm' input file\n");
		exit(1);
	}
	fscanf(ref_f, "%s\n", temp);
	fscanf(ref_f, "%d %d\n", &x, &y);
	fscanf(ref_f, "%d\n",&s);
	if ((x != y) && (x != IMAGE_DIM)){
		fprintf(stderr, "Error opening 'task2_corrected_output.ppm' input file\n");
		exit(1);
	}

	for (int y = 0; y < IMAGE_DIM; y++){
		for (int x = 0; x < IMAGE_DIM; x++){
			int i = x + y*IMAGE_DIM;
			fread(&ref_image[i], sizeof(unsigned char), 3, ref_f);
		}
	}

	fclose(ref_f);

	int errors = 0;
	for (unsigned int idx = 0; idx< IMAGE_DIM * IMAGE_DIM; idx ++){

		if (image[idx].x != ref_image[idx].x || image[idx].y != ref_image[idx].y || image[idx].z != ref_image[idx].z){
			printf("Mismatch at Index: %d\n", idx);
			printf("Reference \t Mine:\n");
            printf("\tR Vals: %d | %d\n",ref_image[idx].x,  image[idx].x);
            printf("\tR Vals: %d | %d\n",ref_image[idx].y,  image[idx].y);
            printf("\tR Vals: %d | %d\n",ref_image[idx].z,  image[idx].z);
			errors++;
		}
	}
	printf("Found %d errors out of %d pixels\n", errors, IMAGE_DIM*IMAGE_DIM);

}

void cpu_image_blur_A(uchar3 *image, uchar3 *image_output) {
	// Add your implementation here
    int pr = 0;
    int pg = 0;
    int pb = 0;
	 for (int y=0; y<IMAGE_DIM; y++){
        for (int x=0; x<IMAGE_DIM; x++){
            
            pr = 0;
            pg = 0;
            pb = 0;
            for (int ry = -1; ry < 2; ry++){
                int y_adj = y + ry;

                if (y_adj < 0){
                    y_adj += IMAGE_DIM;
                }

                if (y_adj >= IMAGE_DIM){
                    y_adj -= IMAGE_DIM;
                }

                for (int rx = -1; rx < 2; rx++){
                
                    int x_adj = x + rx;
                    if (x_adj < 0){
                        x_adj += IMAGE_DIM;
                    }

                    if (x_adj >= IMAGE_DIM){
                        x_adj -= IMAGE_DIM;
                    }

                    uchar3 pixel = image[x_adj + IMAGE_DIM*y_adj];

                    pr += pixel.x;
                    pg += pixel.y;
                    pb += pixel.z;
                }
            }
            pr = pr / 9;
            pg = pg / 9;
            pb = pb / 9;
            image_output[x + IMAGE_DIM * y] = make_uchar3((uchar)pr, (uchar)pg, (uchar)pb);
        }
     }
}

void cpu_image_blur(uchar3 *image, uchar3 *image_output, int r) {
	// Add your implementation here
    int pr = 0;
    int pg = 0;
    int pb = 0;
	 for (int y=0; y<IMAGE_DIM; y++){
        for (int x=0; x<IMAGE_DIM; x++){
            
            pr = 0;
            pg = 0;
            pb = 0;
            for (int ry = 0-r; ry < 1+r; ry++){
                int y_adj = y + ry;

                if (y_adj < 0){
                    y_adj += IMAGE_DIM;
                }

                if (y_adj >= IMAGE_DIM){
                    y_adj -= IMAGE_DIM;
                }

                for (int rx = 0-r; rx < 1+r; rx++){
                
                    int x_adj = x + rx;
                    if (x_adj < 0){
                        x_adj += IMAGE_DIM;
                    }

                    if (x_adj >= IMAGE_DIM){
                        x_adj -= IMAGE_DIM;
                    }

                    uchar3 pixel = image[x_adj + IMAGE_DIM*y_adj];

                    pr += pixel.x;
                    pg += pixel.y;
                    pb += pixel.z;
                }
            }
            pr = pr / ((2*r + 1) * (2*r + 1));
            pg = pg / ((2*r + 1) * (2*r + 1));
            pb = pb / ((2*r + 1) * (2*r + 1));
            image_output[x + IMAGE_DIM * y] = make_uchar3((uchar)pr, (uchar)pg, (uchar)pb);
        }
     }
}

void valid_CpuGpu(uchar3 *ref_image, uchar3 *test_image) {
	// Add your implementation here
    int errors=0;
    for (int i=0; i < IMAGE_DIM*IMAGE_DIM; i++){
        uchar3 ref_pixel = ref_image[i];
        uchar3 test_pixel = test_image[i];

        if (ref_pixel.x != test_pixel.x || ref_pixel.y != test_pixel.y || ref_pixel.z != test_pixel.z){
            errors++;
            printf("Error at index: %d\n", i);
            printf("Expected at index: %d\n", ref_pixel.x);
            printf("Found at index: %d\n", test_pixel.x);

        }
    }
    printf("Found %d errors\n", errors);

}