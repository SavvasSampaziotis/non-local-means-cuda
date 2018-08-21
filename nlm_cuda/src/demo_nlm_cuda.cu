#include <stdio.h>
#include <stdlib.h>


#include "array_utilities.h"

#include "nlm_cuda.h"


int main(int argc, char** argv)
{	
	int patchSize_H = 7;
	int patchSize_W = 5;

	char* filename;
	if(argc == 2)
	{
		printf("file chosen: %s", argv[1]);
		filename = argv[1];
	}

	int H,W;
	int N; // N=H*W=number of pixels in image
	int M; // M = number of pixels in the irregullar search window

	/*Local Host Pointers*/
	float *image; // H-by-W  Image, stored in continues 1-by-N array (N = H*W)

	/* CUDA-Mememory Pointers*/
	float* d_image; // Memory Container of the 2D H-by-W image (in 1-by-HW 1D array)
	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)
	float* d_distMatrix; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)	
	
	/*TODO: individual d_filtSigma h_ij*/
	// float filtSigma
	// float d_filtSigma


	/* Read Data */
	read_dataset(&H, &W, &image, filename);
	N = H*W;

	M = patchSize_H*patchSize_W;

	/* Allocate Device Memory */
	cudaMalloc( (void**) &d_image, N*sizeof(float) );
	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );

	cudaMalloc( (void**) &d_distMatrix, N*sizeof(float) );

	/* Transfer Data too Device Memory */
	cudaMemcpy(d_image, image, N*sizeof(float), cudaMemcpyHostToDevice);


	// NLM

	// Generate 3D Cube of patches
	dim3 blockDim2D(4,4,1); 
  	dim3 gridDim2D(N/blockDim2D.x, N/blockDim2D.y,1); 
	generate_3D_cube<<< , >>>





	cudaFree(d_image);
	cudaFree(d_patchCube);
	cudaFree(d_distMatrix);
}
