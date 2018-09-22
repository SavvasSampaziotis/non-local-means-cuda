/*
	Header File for the non Local Means Imlementation Host Functions.

	Author: Savvas Sampaziotis
*/
#ifndef NLM_CUDA_H
#define  NLM_CUDA_H

#include "reduction.h"
#include "nlm_cuda_kernels.h"



/* 
	Main Caller: This function invokes the rest of the helper functions step-bu-step, thus realizing the complte NLM Algorithm

	returns: The Calclation Time EXCLUDING the data exchange between the host memory and the device memory 
*/
double nonLocalMeansCUDA(float* image_in, float *image_out, int H,int W, int patch_H, int patch_W, int patchSigmaH,int patchSigmaW, float sigma);



/*
 Helper Functions:
	The functions below are wrappers -kenrel callers with the appropriate grid-block dimensions and arguments. 

	The Implementation of the NLM Algorithmn has been broken down to these 4 steps for readability and convinience in debugging. 
*/

// Generates the patches and applies the Gaussian Mask
void generate_3D_cube(float *d_image, float* d_patchCube, int H, int W, int patch_H, int patch_W, int patchSigmaH, int patchSigmaW);

// Calculates the Dist matrix among all rows of the NxM patch-dataset
void calcDistMatrix(float* d_dist, float* d_patchCube, int N, int M, float sigma);

// Finds the Maximum Value on each row of the Distance Matrix, and substitutes the diagonal with each max-value
void clip_dist_diag(float* d_dist, ReductionCache rc, int N );

// Calculates the weighted average of each image-pixel, by performing matrix multiplication and elemnt-wise division between vectors.
void calc_weighted_average(float *d_dist, float* d_image, ReductionCache rc_dist_srow, ReductionCache rc, int N);



#endif