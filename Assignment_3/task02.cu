#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <time.h>
#include <unistd.h>

#define IMAGE_DIM 2048

/* Kernel Launch Specs */
#define THREADS_PER_BLOCK_X 1024
#define THREADS_PER_BLOCK_Y 1

using uchar = unsigned char;

void output_image_file(uchar* image);
void input_image_file(char* filename, uchar3* image);
void checkCUDAError(const char *msg);
void validate_image_file(uchar* ref_image, uchar* image, uchar* host_array); // Adding this one to see my own outputs
void unwind_arrays(uchar3 *image, uchar *r_channel, uchar *g_channel, uchar *b_channel);

/* NOTES ABOUT CODE: 
 * 
 * - I think everything is working below, because out put image looks correct
 * 		- But there is a slight mismatch using cmp / diff 
 		- It seems to be a slight rounding error in 10k pixels
		- I tried to fix with floor, adding 0.5 to help rounding, but I think the problem is something in how the reference was generated
			differently than the formula I am using 
			
 * - For timing, we use Cuda Events. This is a bit different than task01, but easier in this case.
 * 		- To time both kernels at once, I use 2 events which should be fine since everything is on the default stream
		- But I noticed that the uchar3 version of the kernel was taking a lot longer than the 3 uchar array version, only 
		- When I ran it first. I think the overhead for launching the kernel is rought 2 seconds, so I add a warm up for both.
			- I add the 2nd warm up because I want both kernels to have ran once before timing, to keep things as identical as possible.
 */

/* My Code: */
__global__ void image_to_grayscale(uchar3 *image, uchar *image_output) {
	// Add your implementation here
    unsigned int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
	unsigned int image_idx = tidx + tidy * IMAGE_DIM;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){
        uchar3 pixel = image[image_idx];
		float gray_pixel = floor((0.21f * pixel.x) + (0.72f * pixel.y) + (0.07f * pixel.z));
        image_output[image_idx] = (uchar)(gray_pixel);
    }
}

__global__ void image_to_grayscale_3channel(uchar *image_r, uchar *image_g, uchar * image_b, uchar *image_output) {
	// Add your implementation here
    unsigned int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
	unsigned int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
	unsigned int image_idx = tidx + tidy * IMAGE_DIM;

    if (tidx < IMAGE_DIM && tidy < IMAGE_DIM){
		float gray_pixel = floor((0.21f * image_r[image_idx ]) + (0.72f * image_g[image_idx ]) + (0.07f * image_b[image_idx ]));
        image_output[image_idx] = (uchar)(gray_pixel);
    }
}

/* Host code */

int main(void) {
	unsigned int image_size, image_output_size, image_channel_size;
	uchar3 *d_image, *h_image;
	uchar  *d_image_output, *h_image_output, *d_r_channel, *d_g_channel, *d_b_channel, *h_r_channel, *h_g_channel, *h_b_channel;
	cudaEvent_t start_1channel, stop_1channel, start_3channel, stop_3channel;
	float ms_1 = 0.0f;
	float ms_3 = 0.0f;

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar3);
	image_output_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar);
	image_channel_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar);
	
	// create timers
    // Create the events
    cudaEventCreate(&start_1channel);
    cudaEventCreate(&stop_1channel);
	cudaEventCreate(&start_3channel);
    cudaEventCreate(&stop_3channel);

	// allocate memory on the GPU for the output image
    cudaMalloc((void **)&d_image, image_size);

	cudaMalloc((void **)&d_r_channel, image_channel_size);
	cudaMalloc((void **)&d_g_channel, image_channel_size);
	cudaMalloc((void **)&d_b_channel, image_channel_size);

    cudaMalloc((void **)&d_image_output, image_output_size);
	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar3*)malloc(image_size);
	h_image_output = (uchar*)malloc(image_output_size);

	h_r_channel = (uchar*)malloc(image_channel_size);
	h_g_channel = (uchar*)malloc(image_channel_size);
	h_b_channel = (uchar*)malloc(image_channel_size);

	input_image_file("input.ppm", h_image);

	// Move arrays:
	unwind_arrays(h_image, h_r_channel, h_g_channel, h_b_channel);

	// copy image to device memory
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_channel, h_r_channel, image_channel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_g_channel, h_g_channel, image_channel_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b_channel, h_b_channel, image_channel_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	// launch kernel
    // Setup Kernel Launch Params
	int numBlocksX = (IMAGE_DIM + THREADS_PER_BLOCK_X - 1) / THREADS_PER_BLOCK_X;
	int numBlocksY = (IMAGE_DIM + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;

	dim3 numBlocks(numBlocksX, numBlocksY, 1);
	dim3 numThreads(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1);
	
	// warm up:
	image_to_grayscale <<<numBlocks, numThreads>>>(d_image, d_image_output);
	cudaDeviceSynchronize();

	// Now time 1 Channel
    cudaEventRecord(start_1channel, 0); // Using the default stream (stream 0)
	image_to_grayscale <<<numBlocks, numThreads>>>(d_image, d_image_output);
    
    cudaEventRecord(stop_1channel, 0);
    cudaEventSynchronize(stop_1channel);
    cudaEventElapsedTime(&ms_1, start_1channel, stop_1channel);

	checkCUDAError("Kernel launch");
	cudaDeviceSynchronize();

	// Warm up 3 channel to make it fair
	image_to_grayscale_3channel <<<numBlocks, numThreads>>>(d_r_channel, d_g_channel, d_b_channel, d_image_output);
	cudaDeviceSynchronize();
	// Now run 3 channel kernel

	cudaEventRecord(start_3channel, 0); // Using the default stream (stream 0)
	image_to_grayscale_3channel <<<numBlocks, numThreads>>>(d_r_channel, d_g_channel, d_b_channel, d_image_output);
    cudaEventRecord(stop_3channel, 0);
    cudaEventSynchronize(stop_3channel);
    cudaEventElapsedTime(&ms_3, start_3channel, stop_3channel);

	// copy the image back from the GPU for output to file
    cudaMemcpy(h_image_output, d_image_output, image_output_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution time:\n");
	printf("RGB Array:\t%f\n", ms_1);
	printf("3  Arrays:\t%f\n", ms_3);

	// output image
	output_image_file(h_image_output);
	
	// validate_image_file(ref_out, test_out, h_image_output);

	//cleanup
    cudaEventDestroy(start_1channel);
    cudaEventDestroy(stop_1channel);
	cudaEventDestroy(start_3channel);
    cudaEventDestroy(stop_3channel);

    cudaFree(d_image); cudaFree(d_image_output);
    free(h_image); free(h_image_output);

	cudaFree(d_r_channel); cudaFree(d_g_channel); cudaFree(d_b_channel);
	free(h_r_channel); free(h_g_channel); free(h_b_channel);

	
	
	return 0;
}

void output_image_file(uchar* image)
{
	FILE *f; //output file handle

	//open the output file and write header info for PPM filetype
	const char * input_file = "output.ppm";
	f = fopen(input_file, "wb");
	if (f == NULL){
		fprintf(stderr, "Error opening 'output.ppm' output file\n");
		exit(1);
	}
	fprintf(f, "P5\n"); //grayscale PPM file type
	fprintf(f, "%d %d\n%d\n", IMAGE_DIM, IMAGE_DIM, 255);
	for (int y = 0; y < IMAGE_DIM; y++){
		for (int x = 0; x < IMAGE_DIM; x++){
			int i = x + y*IMAGE_DIM;
			fwrite(&image[i], sizeof(unsigned char), 1, f); //only write garyscale
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


void validate_image_file(uchar* ref_image, uchar* image, uchar* host_array)
{
	FILE *f; //input file handle
	char temp[256];
	unsigned int x, y, s;

	//open the output file and write header info for PPM filetype
	f = fopen("output.ppm", "rb");
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
			fread(&image[i], sizeof(unsigned char), 1, f);
		}
	}

	fclose(f);

	FILE *ref_f; //input file handle
	//open the output file and write header info for PPM filetype
	ref_f = fopen("task02_correct_output.ppm", "rb");
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
			fread(&ref_image[i], sizeof(unsigned char), 1, ref_f);
		}
	}

	fclose(ref_f);

	int errors = 0;
	for (unsigned int idx = 0; idx< IMAGE_DIM * IMAGE_DIM; idx ++){
		if (image[idx] != ref_image[idx]){
			printf("Mismatch at Index: %d\n", idx);
			printf("\tRef Val: %f\n", (float)ref_image[idx]);
			printf("\tMy  Val: %f\n", (float)image[idx]);
			printf("\tHA  Val: %f\n", (float)host_array[idx]);
			errors++;
		}
	}
	printf("Found %d errors out of %d pixels\n", errors, IMAGE_DIM*IMAGE_DIM);

}

void unwind_arrays(uchar3 *image, uchar *r_channel, uchar *g_channel, uchar *b_channel){
	for (unsigned int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++){
		uchar3 pixel = image[i];
		r_channel[i] = pixel.x;
		g_channel[i] = pixel.y;
		b_channel[i] = pixel.z;
	}

}