#include <stdio.h>
#include <stdlib.h>

#include "array_utilities.h"
#include "reduction.h"
#include "nlm_cuda.h"

int i=0;

void debug_print(int N, int M, float* d_ptr, const char* fullname)
{
	float *temp = (float*) malloc(M*N*sizeof(float));
	cudaMemcpy(temp, d_ptr, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	// print_array(N,M,temp);

	// printf("%s\n", fullname);
	write_datfile(N,M,temp, fullname);
}

int main(int argc, char** argv)
{
	int patchSize_H = 5;
	int patchSize_W = 5;
	float patchSigmaH = 1.66;
	float patchSigmaW = 1.66;
	float sigma = 0.1f;

	char* filename;
	if(argc == 2)
	{
		printf("file chosen: %s", argv[1]);
		filename = argv[1];
	}
	else
	{
		printf("PLEASE GIVE FILE NAME AS PROGRAM INPUT\n");
		return 0;
	}

	int H,W;
	int N; // N=H*W=number of pixels in image
	int M; // M = number of pixels in the irregullar search window

	/*Local Host Pointers*/
	float *image; // H-by-W  Image, stored in continues 1-by-N array (N = H*W)
	float *image2; //output image

	/* CUDA-Mememory Pointers*/
	float* d_image; // Memory Container of the 2D H-by-W image (in 1-by-HW 1D array)
	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)
	float* d_dist; // Memory Container of the 2D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)	

	// This is the reduction cache used for clipping the diagonal of
	// the distance matrix, and performing the Matrix Calculation between the dist matrix and the image
	ReductionCache rc; 
	ReductionCache rc_dist_srow; 
	
	/*TODO: individual d_filtSigma h_ij*/
	// float filtSigma
	// float d_filtSigma

	printf("%s\n", "savvas");

	/* Read Data */
	read_dataset(&H, &W, &image, filename);
	image2 = (float*) malloc(H*W*sizeof(float));

	printf("%s\n", "savvas");

	N = H*W;
	M = patchSize_H*patchSize_W;

	/* Allocate Device Memory */
	cudaMalloc( (void**) &d_image, N*sizeof(float) );
	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	init_reduction_cache(N,N, 32, &rc); 
	init_reduction_cache(N,N, 32, &rc_dist_srow); 

	/* Transfer Data too Device Memory */
	cudaMemcpy(d_image, image, N*sizeof(float), cudaMemcpyHostToDevice);


	/* NLM */
	//Normalise Image Pixel Intensity
	// TODO


	// Generate 3D Cube (2D array of 1xM patches)
	{
		dim3 blockDim2D	( patchSize_W, patchSize_H, 1 ); 
	  	dim3 gridDim2D	( W, H, 1 ); 
		generate_3D_cube<<< gridDim2D, blockDim2D >>> (d_image, d_patchCube, H,W, patchSize_H, patchSize_W);

		debug_print(N, M, d_patchCube, "../data/patchCube.bin");
		
		apply_gaussian_filter<<< gridDim2D, blockDim2D >>>\
			( d_patchCube,  patchSize_H, patchSize_W,  patchSigmaH, patchSigmaW);
	}

	debug_print(N, M, d_patchCube, "../data/patchCube_gaussed.bin");

	// Calculate Distance Matrix between all possible patch-pairs
	{
		int b = 22; // max threads per block: 512 > 22*2=448
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, b, 1 ); 
	  	dim3 gridDim2D	( g, g, 1 ); 
		calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, N, M, sigma);
	}

	debug_print(N, N, d_dist, "../data/dist.bin");
	
	/* Clip Dist Matrix Diagonal */
	{	
		// subtitude each element of the diagonal, with the row-max calculated above.
		int b = 22; // max threads per block: 512 > 22*2=448
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, 1, 1 ); 
	  	dim3 gridDim2D	( g, 1, 1 ); 
		
	  	// clip_dist_diag <<<gridDim2D, blockDim2D >>>(d_dist, 0, N);

		// Find max element in each row
		row_max_WR(N, d_dist, &rc);
		debug_print(N, 1, rc.d_sum, "../data/max_diag.bin");

		// clip_dist_diag <<<gridDim2D, blockDim2D >>>(d_dist, rc.d_sum, N);
	}
	debug_print(N, N, d_dist, "../data/distClipped.bin");
	


	/* Calculate Filtered Image (Weighted Average of Original Image)*/
	// Calculate the Denominator: The sum of each row of the d_dist Matrix
  	row_sum_WR(N, d_dist, &rc_dist_srow);

  	debug_print(N, 1, rc_dist_srow.d_sum, "../data/dist_rowsum.bin");

	// Perform Matrix Multiplication 
	{	
		// Matrix-Vector Multiplication: step 1
		int b = 22; // max threads per block: 512 > 22^2=448
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, b, 1 ); 
	  	dim3 gridDim2D	( g, g, 1 ); 
	 	multi_mat_vector_row<<<gridDim2D, blockDim2D>>>( d_dist, image, /*out*/ d_dist, N);
	 
	  	// Matrix-Vector Multiplication: step 2
	  	row_sum_WR(N, d_dist, &rc);

	  	debug_print(N, 1, rc.d_sum, "../data/dist_image_mult.bin");
	}  	
	// Normalise the Average
	{	
		int b = 256; // max threads per block: 512 
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, 1, 1 ); 
	  	dim3 gridDim2D	( g, 1, 1 ); 
	
  		// div_vector<<<gridDim2D, blockDim2D>>>( rc.d_sum, rc_dist_srow.d_sum, /*out*/ d_image, N);
	}  


	/* Get Image from Device Memory */
	cudaMemcpy(image2, rc.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);

	// print_array(H,W,image2);
	write_datfile(H,W,image2, "../data/filtered_image.bin");

	cudaFree(d_image);
	cudaFree(d_patchCube);
	cudaFree(d_dist);

	free(image);
	free(image2);
}