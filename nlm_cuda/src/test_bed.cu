/*
	Source File for unit-testing each nlm kernel function

	Author: Savvas Sampaziotis
*/


#include "array_utilities.h"
#include <stdio.h>
#include <stdlib.h>

#include "nlm_cuda.h"
#include "reduction.h"



int H,W;
float *data;

void test_generate_3d_cube();
void test_apply_gaussianfilt();
bool test_calc_dist(int N);
bool calc_sum_reduction(int N);
bool test_clip_diag(int N);
bool test_matrix_mult(int N);

void printVerdict(bool p)
{
	if(p)
		printf("Verdict = PASSED\n");
	else
		printf("Verdict = FAILED\n");
}

int main(int argc, char** argv)
{
	
	int N;
	if(argc == 2)
		N = atoi(argv[1]);
	else
		N = 10;

	printf("\nProblem Size N = %d\n",N);


	/***** Test Area *****/
	bool p;
	// p = test_generate_3d_cube();	

	// p = test_apply_gaussianfilt();

	// Calcs the Row Sum of a [NxN] matrix  
	printf("\n------------------------------\n");
	printf("Calc Sum Reduction\n");
	p = calc_sum_reduction(N);
	printVerdict(p);
	
	// Square Matrix A:[NxN]
	printf("\n------------------------------\n");
	printf("Clip Diagonal\n");
	p = test_clip_diag(N);
	printVerdict(p);

	// y=Ax -> A:[NxN] y,x:[Nx1]
	printf("\n------------------------------\n");
	printf("Matrix Multiplication\n");
	p = test_matrix_mult(N);
	printVerdict(p);

	// Calc the NxN dist Matrix of a NxM Data set. M is set constant M=5*5=25
	printf("\n------------------------------\n");
	printf("Calc Distance Matrix\n");
	p = test_calc_dist(N);
	printVerdict(p);
	return 0;
}

bool test_matrix_mult(int N)
{
	/* y = A*x */
	float *A = (float*) malloc(N*N*sizeof(float));
	float *x = (float*) malloc(N*sizeof(float));

	float *d_A, *d_x;
	cudaMalloc((void**) &d_A, N*N*sizeof(float));
	cudaMalloc((void**) &d_x, N*sizeof(float));

	ReductionCache rc;
	init_reduction_cache(N,N, 32, &rc);

	/* Init Data */
	for(int i=0; i<N; i++)
		x[i] = (i%2)?1:-1;
	
	for(int i=0; i<N*N; i++)
		A[i] = i+1;

	cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

	/* Perform Multiplication */
	int b = 20;
	int g = ceil(N/b) + ((N%b==0)?0:1);
	dim3 blockDim2D	( b, b, 1 ); 
  	dim3 gridDim2D	( g, g, 1 ); 
 	multi_mat_vector_row<<<gridDim2D, blockDim2D>>>( d_A, d_x, /*out*/ d_A, N);

  	row_sum_WR(N, d_A, &rc);

  	// Check result
  	float *y = (float*) malloc(N*N*sizeof(float));
  	cudaMemcpy(y, rc.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);

  	float *y2 = (float*) malloc(N*N*sizeof(float));
 	for(int i=0; i<N; i++)
 	{	
 		y2[i] = 0;
 		for (int j = 0; j < N; j++)
 				y2[i] += A[i*N+j]*x[j];
 	}
  	bool passed = true;
  	float MSE = 0;
 	for(int i=0; i<N; i++)
	{
		passed = passed && (y2[i] == y[i]);
 		MSE += (y2[i] - y[i])*(y2[i] - y[i]);
 	}
 	MSE = MSE/N;
	if (passed == false)
	 	printf("MSE=%f\n",MSE);

 	// print_array(1,N,y);
 	// print_array(1,N,y2);

 	// Clean up
	free(A);
	free(x);
	free(y);
	free(y2);
	cudaFree(d_A);
	cudaFree(d_x);
	delete_reduction_cache(&rc);

	return passed;
}


bool test_clip_diag(int N)
{
	
	float *testArray = (float*) malloc(N*N*sizeof(float));
	for(int i=0; i<N*N; i++)
		testArray[i] = i+1; 
	float *d_dist;
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	cudaMemcpy(d_dist, testArray, N*N*sizeof(float), cudaMemcpyHostToDevice);

	ReductionCache rc;
	init_reduction_cache(N,N,32,&rc);

	// Find Max in each row
	row_max_WR(N, d_dist, &rc);
	
	// Replace diag with max
	int b = 22;
	int g = ceil(N/b) + ((N%b==0)?0:1);
	dim3 blockDim2D	( b, 1, 1 ); 
  	dim3 gridDim2D	( g, 1, 1 ); 
	clip_dist_diag <<<gridDim2D, blockDim2D >>>(d_dist, rc.d_sum, N);

	// Test if correct
	float* result = (float*) malloc(N*N*sizeof(float));
	cudaMemcpy(result, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++)
	{	
		testArray[i*N+i] = (i+1)*N;
		// float max = 0;
		// for(int j=0; j<N; j++)
		// 	testArray[i*N+i+1] = (max > testArray[j*N+j+1])? max: testArray[j*N+j+1];
	}

	bool passed = true;
	for(int i=0; i<N*N; i++)
			passed = passed && (testArray[i]==result[i]);

	// printf("Test %s\n", passed ? "PASSED":"FAILED" );

	free(testArray);
	cudaFree(d_dist);
	delete_reduction_cache(&rc);

	return passed;
}

bool calc_sum_reduction(int N)
{
	float *testArray = (float*) malloc(N*N*sizeof(float));
	for(int i=0; i<N*N; i++)
		testArray[i] = i; 
	float *d_dist;
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	cudaMemcpy(d_dist, testArray, N*N*sizeof(float), cudaMemcpyHostToDevice);


	ReductionCache rc;
	init_reduction_cache(N,N,32,&rc);

	row_sum_WR(N, d_dist, &rc);
	// row_max_WR(N, d_dist, &rc);

	float *rsum = (float*) malloc(N*sizeof(float));
	cudaMemcpy(rsum, rc.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);

  	// **** Check Result **** //
  	float MSE = 0;
  	bool passed = true;
  	for(int i=0; i<N; i++)
	{	
		float sum = 0;
		for(int j=0; j<N; j++)
			sum += testArray[i*N+j];
			// sum = (sum>testArray[i*N+j])? sum:testArray[i*N+j];
	
		float e = (sum - rsum[i]);
		passed = passed && (sum==rsum[i]);
		MSE+=e*e;
	}
	MSE = MSE/N;
	if (passed == false)
		printf("MSE=%f\n",MSE);
	
	free(rsum);
	free(testArray);
	cudaFree(d_dist);
	delete_reduction_cache(&rc);

	return passed;
}

bool test_calc_dist(int N)
{
	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)
	float* d_dist; // Memory Container of the 2D HW-by-HW dist_matrix of all patches

	int M = 25;
	float sigma = 0.1;

	// Construct Test 3D Cube 
	float* testArray = (float*) malloc(M*N*sizeof(float));
	for(int i=0; i<N*M; i++)
		testArray[i] = i;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	
	//////////////
	int b = 22;
	int g = ceil(N/b) + ((N%b==0)?0:1);
	dim3 blockDim2D	( b, b, 1 ); 
  	dim3 gridDim2D	( g, g, 1 ); 
	calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, N, M, sigma);

	////////////// Print Results	
	// if (N<=10)
	// print_array(N,M,testArray);
	
	float* dist = (float*) malloc(N*N*sizeof(float)) ;	
	cudaMemcpy(dist, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	// if (N<=10)
	// print_array(N, N, dist);
	// else
		// printf("Matrix to big to print in console\n");

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

	bool passed = (MSE==0);
	if (passed == false)
		printf("MSE=%lf\n",MSE);
	
	
	//// Clean Up
	cudaFree(d_dist);
	cudaFree(d_patchCube);
	free(dist);

	return passed;
}


// TODO: make it to return bool
void test_apply_gaussianfilt(int N)
{	
	float patchSigmaH = 1.66;
	float patchSigmaW = 1.66;
	int pH = 5, pW = 5; 
	int M = pH*pW;

	float* d_patchCube; // Memory Container of the 3D H-by-W-by-patchSize cube containing the (in 1-by-HW 1D array)

	float* testArray = (float*) malloc(M*N*sizeof(float));
	for(int i=0; i<M*N; i++)
		testArray[i] = 1;

	cudaMalloc( (void**) &d_patchCube, M*N*sizeof(float) );
	cudaMemcpy(d_patchCube, testArray, M*N*sizeof(float), cudaMemcpyHostToDevice);
	
	
	//////////////
	dim3 blockDim2D	( pW, pH, 1 ); 
  	dim3 gridDim2D	( N, 1, 1 ); 
	apply_gaussian_filter<<< gridDim2D, blockDim2D >>>( d_patchCube,  pH, pW,  patchSigmaH, patchSigmaW);


	////////////// Print Results
	float* patchCube = (float*) malloc(M*N*sizeof(float)) ;	
	cudaMemcpy(patchCube, d_patchCube, M*N*sizeof(float), cudaMemcpyDeviceToHost);

	// printf("pHxpW Patches:\n");
	// print_array(1, M, patchCube);

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