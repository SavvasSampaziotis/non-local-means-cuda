/*
	Header File for non LOcal Means Imlementation

	Author: Savvas Sampaziotis
*/
#ifndef NLM_CUDA_H
#define  NLM_CUDA_H



/*
	Fills the patchCube 3D array: A 2D array of 1D arrays containing the M-pixels of the patch window.

	The function is indended for a 2D 
	of the image is 'symmetric'
*/
__global__
void generate_3D_cube(float *d_image, float *d_patchCube, int patchSize_W, int patchSize_H);













#endif
