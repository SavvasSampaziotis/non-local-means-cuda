/*
	Header File for non LOcal Means Imlementation Cuda-Kernel Funcrions

	Author: Savvas Sampaziotis
*/
#ifndef NLM_CUDA_KERNELS_H
#define  NLM_CUDA_KERNELS_H

#include "reduction.h"
#include <float.h>

/*
	Fills the patchCube 3D array: A 2D array of 1D arrays containing the M-pixels of the patch window.

	It support non-square patchWindows as well.
	Patches outside the image boundary are handled with zero-padding.

	For an image [W,H]m this kernel function is called with: 
		+ a 2D [WxH] grid of block 
		+ a 2D [pWxpH] block of threads
	Maximum Image dimensions:  65535x65535
	Maximum Patch dimensions:  512x512

	WARNING: patch window must have odd x and y dimensions (1,3,5...7)

*/
__global__
void generate_3D_cube(float *d_image, float *d_patchCube, int H, int W,  int pH, int pW);


/*
	Applies Gaussian Mask on each patch. 

	The kernel is called with a
		+ a 2D [WxH] grid of block
		+ a 2D [pWxpH] block of threads
*/
__global__
void apply_gaussian_filter(float *d_patchCube, int pH, int pW, float patchSigma_h, float patchSigma_w);


/*
	Calcs the distance matrix of all the stuff
*/
__global__
void calc_dist_matrix(float *d_distMatrix, float *d_patchCube, int N, int M, float sigma);



/*

*/
__global__
void calc_dist_matrix_SHARED(float *d_distMatrix, float *d_patchCube,  float sigma);


__global__
void calc_dist_matrix_adaptive(float *d_distMatrix, float *d_patchCube, int N, int M, float* d_hMat);

/*
	Replaces the diagonal of square matrix d_dist with elements of vector d_diag

	Invoked with 1-D grid and 1-D blocks, matching the length of the d_diag vactor and higher
*/
__global__
void clip_dist_diag(float* d_dist, float* d_diag, int N);



__global__
void multi_mat_vector_row(float* d_A, float* d_x, /*out*/ float* d_B, int N);

__global__
void div_vector(float* d_A, float* d_x, /*out*/ float* d_B, int N);

#endif
