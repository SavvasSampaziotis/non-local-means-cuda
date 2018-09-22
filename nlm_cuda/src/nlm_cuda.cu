/*
	Source File for non local means library (host functions)

	Author: Savvas Sampaziotis
*/

#include "nlm_cuda.h"

#include "time_measure.h"

double nonLocalMeansCUDA(float* image_in, float *image_out, int H, int W, \
	int patch_H, int patch_W, int patchSigmaH, int patchSigmaW, float sigma)
{
	TimeInterval calcTime;
	
	// Input Data Size 
	int N = H*W; // number of pixels in image
	int M = patch_H*patch_W; //M = number of pixels in the irregullar search window

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
	
	/* Allocate Device Memory */
	cudaMalloc( (void**) &d_image, N*sizeof(float) );
	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	init_reduction_cache(N,N, 32, &rc); 
	init_reduction_cache(N,N, 32, &rc_dist_srow); 


	/* Transfer Data too Device Memory */
	cudaMemcpy(d_image, image_in, N*sizeof(float), cudaMemcpyHostToDevice);


	/* NLM */
	//Normalise Image Pixel Intensity
	// TODO

	tic(&calcTime);

	generate_3D_cube(d_image, d_patchCube, H, W,  patch_H,  patch_W,  patchSigmaH, patchSigmaW);

 	calcDistMatrix(d_dist, d_patchCube, N, M, sigma);

	clip_dist_diag(d_dist, rc, N);

	calc_weighted_average(d_dist,  d_image, rc_dist_srow, rc, N);

	toc(&calcTime);

	/* Get Image from Device Memory */
	cudaMemcpy(image_out, d_image, N*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_image);
	cudaFree(d_patchCube);
	cudaFree(d_dist);
	delete_reduction_cache(&rc_dist_srow);
	delete_reduction_cache(&rc);

	return calcTime.seqTime;
}

void generate_3D_cube(float *d_image, float* d_patchCube, int H, int W, int patch_H, int patch_W, int patchSigmaH, int patchSigmaW)
{
	// Generate 3D Cube (2D array of 1xM patches)
	dim3 blockDim2D	( patch_W, patch_H, 1 ); 
  	dim3 gridDim2D	( W, H, 1 ); 

	generate_3D_cube<<< gridDim2D, blockDim2D >>> (d_image, d_patchCube, H,W, patch_H, patch_W);
	apply_gaussian_filter<<< gridDim2D, blockDim2D >>>\
		( d_patchCube,  patch_H, patch_W,  patchSigmaH, patchSigmaW);

}


void calcDistMatrix(float* d_dist, float* d_patchCube, int N, int M, float sigma)
{
	// Calculate Distance Matrix between all possible patch-pairs
	int b = 22; // max threads per block: 512 > 22*2=448
	int g = ceil(N/b) + ((N%b==0)?0:1);
	dim3 blockDim2D	( b, b, 1 ); 
  	dim3 gridDim2D	( g, g, 1 ); 
	calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, N, M, sigma);
}

void clip_dist_diag(float* d_dist, ReductionCache rc, int N )
{	
	/* Clip Dist Matrix Diagonal */
	int b = 256; // max threads per block: 512 > 22*22=448
	int g = ceil(N/b) + ((N%b==0)?0:1);
	dim3 blockDim2D	( b, 1, 1 ); 
  	dim3 gridDim2D	( g, 1, 1 ); 
	
  	clip_dist_diag <<<gridDim2D, blockDim2D >>>(d_dist, 0, N);

	// Find max element in each row
	row_max_WR(N, d_dist, &rc);

	// subtitude each element of the diagonal, with the row-max calculated above.
	clip_dist_diag <<<gridDim2D, blockDim2D >>>(d_dist, rc.d_sum, N);
}

void calc_weighted_average(float *d_dist, float* d_image, ReductionCache rc_dist_srow, ReductionCache rc, int N)
{
	// Calculate the Denominator: The sum of each row of the d_dist Matrix
  	row_sum_WR(N, d_dist, &rc_dist_srow);

	// Perform Matrix Multiplication 
	{	
		// Matrix-Vector Multiplication: step 1
		int b = 22; // max threads per block: 512 > 22^2=448
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, b, 1 ); 
	  	dim3 gridDim2D	( g, g, 1 ); 
	 	multi_mat_vector_row<<<gridDim2D, blockDim2D>>>( d_dist, d_image, /*out*/ d_dist, N);
	 
	  	// Matrix-Vector Multiplication: step 2
	  	row_sum_WR(N, d_dist, &rc);
	}  	
	// Normalise the Average
	{	
		int b = 256; // max threads per block: 512 
		int g = ceil(N/b) + ((N%b==0)?0:1);
		dim3 blockDim2D	( b, 1, 1 ); 
	  	dim3 gridDim2D	( g, 1, 1 ); 
	
  		div_vector<<<gridDim2D, blockDim2D>>>( rc.d_sum, rc_dist_srow.d_sum, /*out*/ d_image, N);
	}  
}