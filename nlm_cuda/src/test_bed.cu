

#include "array_utilities.h"
#include <stdio.h>
#include <stdlib.h>

#include "nlm_cuda.h"


int H,W;
float *data;

void test_generate_3d_cube();
void test_apply_gaussianfilt();


int main(int argc, char** argv)
{
	
	char* filename;
	if(argc == 2)
		filename = argv[1];
	else
		printf("NO ARGS GIVEN\n");
	printf("[MAIN]:\t.dat File targeted for denoising: %s", filename);
	read_dataset(&H, &W, &data, filename);
	printf("%f\n\n",data[3]);


	/***** Test Area *****/

	// test_generate_3d_cube();	

	test_apply_gaussianfilt();
	// 

	// Free data
	free(data);
	return 0;
}




void test_generate_3d_cube(){
	
	printf("Generate 3D Cube\n" );

	float* d_dist; // Memory Container of the 2D H-by-W image (in 1-by-HW 1D array)
	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)

	int H = 4, W = 5;
	int N = H*W;
	int pH = 3, pW=3; 
	int M = pH*pW;

	float* testArray = (float*) malloc(N*sizeof(float));
	for(int i=0; i<N*M; i++)
		testArray[i] = i+1;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc( (void**) &d_dist, N*sizeof(float) );
	
	//////////////
	dim3 blockDim2D	( pW, pH, 1 ); 
  	dim3 gridDim2D	( W, H, 1 ); 
	calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, H,W, pH,pW);


	////////////// Print Results	
	printf("Test Array:\n");
	print_array(H,W,testArray);
	
	float* dist = (float*) malloc(N*N*sizeof(float)) ;	
	cudaMemcpy(dist, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	if (M < 10)
	{
		printf("pHxpW Patches:\n");
		print_array(H*W, pH*pW, patchCube);
	}
	else
		printf("Array too big to print in console\n");

	//// Clean Up
	cudaFree(d_dist);
	cudaFree(d_patchCube);
	free(dist);
}




void test_apply_gaussianfilt(){
	
	printf("Apply Gaussian Filter on each patch of the 3D Cube\n" );

	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)

	float patchSigma = 1;
	int pH = 7, pW = 5; 
	
	int M = pH*pW;
	int H = 4, W = M;
	int N = H*W;

	float* testArray = (float*) malloc(M*N*sizeof(float));
	for(int i=0; i<M*N; i++)
		testArray[i] = 1;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	
	//////////////
	dim3 blockDim2D	( pW, pH, 1 ); 
  	dim3 gridDim2D	( W, H, 1 ); 
	
	apply_gaussian_filter<<< gridDim2D, blockDim2D >>>( d_patchCube,  pH, pW,  1, 2);


	////////////// Print Results
	
	printf("\nTest Array:");
	print_array(N,M,testArray);

	float* patchCube = (float*) malloc(M*N*sizeof(float)) ;	
	cudaMemcpy(patchCube, d_patchCube, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	printf("\npHxpW Patches:");
	print_array(3, M, patchCube);


	//// Clean Up
	cudaFree(d_patchCube);
	free(patchCube);
}



void test_generate_3d_cube(){
	
	printf("Generate 3D Cube\n" );


	float* d_image; // Memory Container of the 2D H-by-W image (in 1-by-HW 1D array)
	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)

	int H = 4, W = 5;
	int N = H*W;
	int pH = 3, pW=3; 
	int M = pH*pW;

	float* testArray = (float*) malloc(N*sizeof(float));
	for(int i=0; i<N; i++)
		testArray[i] = i+1;

	cudaMalloc( (void**) &d_image, N*sizeof(float) );
	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_image, testArray, N*sizeof(float), cudaMemcpyHostToDevice);
	
	
	//////////////
	dim3 blockDim2D	( pW, pH, 1 ); 
  	dim3 gridDim2D	( W, H, 1 ); 
	generate_3D_cube<<< gridDim2D, blockDim2D >>> (d_image, d_patchCube, H,W, pH,pW);


	////////////// Print Results
	if (W < 10)
	{
		printf("Test Array:\n");
		print_array(H,W,testArray);
	}
	else
		printf("Array too big to print in console\n");

	float* patchCube = (float*) malloc(M*N*sizeof(float)) ;	
	cudaMemcpy(patchCube, d_patchCube, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	if (M < 10)
	{
		printf("pHxpW Patches:\n");
		print_array(H*W, pH*pW, patchCube);
	}
	else
		printf("Array too big to print in console\n");

	//// Clean Up
	cudaFree(d_image);
	cudaFree(d_patchCube);
	free(patchCube);
}