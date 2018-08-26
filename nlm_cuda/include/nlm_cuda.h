/*
	Header File for non LOcal Means Imlementation

	Author: Savvas Sampaziotis
*/
#ifndef NLM_CUDA_H
#define  NLM_CUDA_H



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



#endif
