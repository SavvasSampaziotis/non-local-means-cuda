

#include "array_utilities.h"
#include <stdio.h>
#include <stdlib.h>

#include "nlm_cuda.h"
#include "reduction.h"
#include "time_measure.h"
int H,W;
float *data;

void test_generate_3d_cube();
void test_apply_gaussianfilt();
void test_calc_dist();
void compare_calcdist();
void calc_sum_reduction();

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

	// test_apply_gaussianfilt();

	// test_calc_dist();
	
	// compare_calcdist();

	calc_sum_reduction();

	// Free data
	free(data);
	return 0;
}
void calc_sum_reduction()
{
	int N = 100;
	
	float *testArray = (float*) malloc(N*N*sizeof(float));
	for(int i=0; i<N*N; i++)
		testArray[i] = i+1; 
	float *d_dist;
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	cudaMemcpy(d_dist, testArray, N*N*sizeof(float), cudaMemcpyHostToDevice);

/*
	float *d_rsum_lvl1;
	cudaMalloc( (void**) &d_rsum_lvl1, N*Bx*sizeof(float) );

	float *d_rsum;
	cudaMalloc( (void**) &d_rsum, N*sizeof(float) );
	
	int blockNum = exp2f(floor(log2f(N/Bx)));
	// Level 2 Reduction: N/Bx Blocks
	{
		dim3 blockDim2D	( Bx, 1, 1 ); 
	  	dim3 gridDim2D	( blockNum, N, 1 ); 
	  	rowsum<<<gridDim2D,blockDim2D>>>(N, d_dist, d_rsum_lvl1);
	  	// rowmax<<<gridDim2D,blockDim2D>>>(N, d_dist, d_rsum_lvl1);
	 
	}
	// Level 1 Reduction: 1 Block remaining to reduct.
	{
		dim3 blockDim2D	( blockNum,1, 1 ); 
	  	dim3 gridDim2D	( 1, N, 1 ); 
	  	rowsum<<<gridDim2D,blockDim2D>>>(N, d_dist, d_rsum);
	  	// rowmax<<<gridDim2D,blockDim2D>>>(N, d_dist, d_rsum);
	}
*/
	ReductionCache rc;
	init_reduction_cache(N,N,16,&rc);

	// row_sum_WR(N, d_dist, &rc);
	row_max_WR(N, d_dist, &rc);

	float *rsum = (float*) malloc(N*sizeof(float));
	cudaMemcpy(rsum, rc.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);

  	// **** Check Result **** //
  	float MSE = 0;
  	for(int i=0; i<N; i++)
	{	
		float sum = 0;
		for(int j=0; j<N; j++)
			// sum+=testArray[i*N+j];
			sum = (sum>testArray[i*N+j])? sum:testArray[i*N+j];
		// printf("\t%f\n", sum );		
		float e = (sum - rsum[i]);
		MSE+=e*e;
	}
	MSE = MSE/N;
	// print_array(N,1,rsum);
	printf("MSE = %f\t%s\n", MSE, (MSE==0)? "PASSED":"FAILED" );
	
	free(rsum);
	free(testArray);
	cudaFree(d_dist);
	delete_reduction_cache(&rc);
}

void test_calc_dist(){
	
	printf("Test Dist Matrix Calculation of N vectors of length M\n" );

	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)
	float* d_dist; // Memory Container of the 2D HW-by-HW dist_matrix of all patches

	int N = 10;
	int M = 4;
	float sigma = 10;

	// Construct Test 3D Cube 
	float* testArray = (float*) malloc(M*N*sizeof(float));
	for(int i=0; i<N*M; i++)
		testArray[i] = i+1;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	
	//////////////
	int b = 50;
	int g = ceil(N/b);
	dim3 blockDim2D	( b, b, 1 ); 
  	dim3 gridDim2D	( g, g, 1 ); 
	calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, N, M, sigma);
	

	dim3 blockDim2D_SH	( 2*M, 1, 1 ); 
  	dim3 gridDim2D_SH	( N, N, 1 ); 
	calc_dist_matrix_SHARED<<< gridDim2D_SH, blockDim2D_SH >>> (d_dist, d_patchCube, sigma);

	////////////// Print Results	
	printf("Test Array:\n");
	if (N<=10)
		print_array(N,M,testArray);
	
	float* dist = (float*) malloc(N*N*sizeof(float)) ;	
	cudaMemcpy(dist, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	if (N<=10)
		print_array(N, N, dist);
	else
		printf("Matrix to big to print in console\n");
// print_array(N, N, dist);
	////////////// Check results
	// Check if brute force result is equal to that of the Cuda kernel 
	double MSE = 0;
	for(int i=0; i<N; i++)
		for(int j=0; j<N; j++)
		{
			float D=0;
			for(int m=0; m<M; m++)
			{
				float a,b;
				a = testArray[i*M+m];
				b = testArray[j*M+m]; 	
				D += (a-b)*(a-b);
			}
			D = exp(-D/sigma/sigma);
			MSE +=  (dist[i*N+j]-D)*(dist[i*N+j]-D);
		}
	MSE = MSE/N/N;	
	printf("Mean Square Error = %lf\n", MSE);
	
	//// Clean Up
	cudaFree(d_dist);
	cudaFree(d_patchCube);
	free(dist);
}



void compare_calcdist()
{
	printf("Test Dist Matrix Calculation of N vectors of length M\n" );

	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)
	float* d_dist; // Memory Container of the 2D HW-by-HW dist_matrix of all patches

	int N = 500;
	int M = 40;
	float sigma = 10;

	// Construct Test 3D Cube 
	float* testArray = (float*) malloc(M*N*sizeof(float));
	for(int i=0; i<N*M; i++)
		testArray[i] = i+1;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	
	//////////////
	int K = 100;

	TimeInterval ti;
	tic(&ti);
	for(int i=0; i<K; i++)
	{
		dim3 blockDim2D	( 2*M, 1, 1 ); 
	  	dim3 gridDim2D	( N, N, 1 ); 
		calc_dist_matrix_SHARED<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, sigma);
	}
	double T1 = toc(&ti);
	T1 = T1;

	tic(&ti);
	for(int i=0; i<K; i++)
	{

		dim3 blockDim2D	( 1, 1, 1 ); 
	  	dim3 gridDim2D	( N, N, 1 ); 
		calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube,N, M, sigma);
	}
	double T2 = toc(&ti);
	T2 = T2;
	
	printf("Shared = %f sec \t Simple = %f sec\n", T1, T2 );
	//// Clean Up
	cudaFree(d_dist);
	cudaFree(d_patchCube);
}



void test_apply_gaussianfilt(){
	
	printf("Apply Gaussian Filter on each patch of the 3D Cube\n" );

	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)

	float patchSigmaH = 1;
	float patchSigmaW = 2;
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
	
	apply_gaussian_filter<<< gridDim2D, blockDim2D >>>( d_patchCube,  pH, pW,  patchSigmaH, patchSigmaW);


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