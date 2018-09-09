

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
void test_clip_diag();
void test_matrix_mult();

int main(int argc, char** argv)
{
	
	// char* filename;
	// if(argc == 2)
	// 	filename = argv[1];
	// else
	// 	printf("NO ARGS GIVEN\n");
	// printf("[MAIN]:\t.dat File targeted for denoising: %s", filename);
	// read_dataset(&H, &W, &data, filename);
	// printf("%f\n\n",data[3]);


	/***** Test Area *****/

	// test_generate_3d_cube();	

	// test_apply_gaussianfilt();

	// test_calc_dist();
	
	// compare_calcdist();

	calc_sum_reduction();
	
	// test_clip_diag();

	// test_matrix_mult();
	
	// Free data
	// free(data);
	return 0;
}

void test_matrix_mult()
{
	/* y = A*x */

	int N = 100;

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

 	printf("MSE=%f\tTest %s\n",MSE, (passed?"PASSED":"FAILED") );

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
}


void test_clip_diag()
{
	int N = 100;
	
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
	// print_array(N,N,testArray);
	cudaMemcpy(testArray, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	// print_array(N,N,testArray);

	bool passed = true;
	for(int i=0; i<N; i++)
		passed = passed && testArray[i*N+i]==(i+1)*N;

	printf("Test %s\n", passed ? "PASSED":"FAILED" );
	


	free(testArray);
	cudaFree(d_dist);
	delete_reduction_cache(&rc);
}

void calc_sum_reduction()
{
	int N = 200;
	
	float *testArray = (float*) malloc(N*N*sizeof(float));
	for(int i=0; i<N*N; i++)
		testArray[i] = i+1; 
	float *d_dist;
	cudaMalloc( (void**) &d_dist, N*N*sizeof(float) );
	cudaMemcpy(d_dist, testArray, N*N*sizeof(float), cudaMemcpyHostToDevice);


	ReductionCache rc;
	init_reduction_cache(N,N,16,&rc);

	row_sum_WR(N, d_dist, &rc);
	// row_max_WR(N, d_dist, &rc);

	float *rsum = (float*) malloc(N*sizeof(float));
	cudaMemcpy(rsum, rc.d_sum, N*sizeof(float), cudaMemcpyDeviceToHost);

  	// **** Check Result **** //
  	float MSE = 0;
  	for(int i=0; i<N; i++)
	{	
		float sum = 0;
		for(int j=0; j<N; j++)
			sum += testArray[i*N+j];
			// sum = (sum>testArray[i*N+j])? sum:testArray[i*N+j];
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

	int N = 4096;
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
	int b = 22;
	int g = ceil(N/b) + ((N%b==0)?0:1);
	printf("grid num = %d\n", g);
	dim3 blockDim2D	( b, b, 1 ); 
  	dim3 gridDim2D	( g, g, 1 ); 
	calc_dist_matrix<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, N, M, sigma);
	

	// dim3 blockDim2D_SH	( 2*M, 1, 1 ); 
 //  	dim3 gridDim2D_SH	( N, N, 1 ); 
	// calc_dist_matrix_SHARED<<< gridDim2D_SH, blockDim2D_SH >>> (d_dist, d_patchCube, sigma);

	////////////// Print Results	
	if (N<=10)
		print_array(N,M,testArray);
	
	float* dist = (float*) malloc(N*N*sizeof(float)) ;	
	cudaMemcpy(dist, d_dist, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	if (N<=10)
		print_array(N, N, dist);
	else
		printf("Matrix to big to print in console\n");

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
		// calc_dist_matrix_SHARED<<< gridDim2D, blockDim2D >>> (d_dist, d_patchCube, sigma);
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
	int pH = 3, pW = 3; 
	
	int M = pH*pW;
	int H = 6, W = 6;
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
	
	printf("Test Array:\n");
	print_array(N,M,testArray);

	float* patchCube = (float*) malloc(M*N*sizeof(float)) ;	
	cudaMemcpy(patchCube, d_patchCube, M*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	printf("pHxpW Patches:\n");
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