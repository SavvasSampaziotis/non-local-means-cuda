
#include "reduction.h"

#include <float.h>

// #define FLT_MIN 1.175494e-38

/*
	Source: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
*/
__device__
int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ 
int getGlobalIdx_2D_3D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) \
					+ (threadIdx.z * (blockDim.x * blockDim.y)) 	\
					+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}


__device__
int symmetric_index_padding(int i, int N)
{
	if ( i < 0 )	
		return -i;
	else if ( i < N )
		return i;
	else
		return N-(i-N)-1;
		// return N-1;
}

__global__
void generate_3D_cube(float *d_image, float *d_patchCube, int H, int W, int pH, int pW)
{
	int k = getGlobalIdx_2D_2D();	
	
	// Local coordinates on patch window (ref pixel at the center)
	int patch_i = threadIdx.y - (pH-1)/2;
	int patch_j = threadIdx.x - (pW-1)/2;
	
	int im_i = blockIdx.y;
	int im_j = blockIdx.x;

 	if ( !( (im_i<H)&&(im_j<W) ) )
 		return;

	int i = im_i+patch_i;
	int j = im_j+patch_j;

	i = symmetric_index_padding(i,H);
	j = symmetric_index_padding(j,W);
	d_patchCube[k] = d_image[ i*W + j ];
	// d_patchCube[k] = i*W + j;


	// if ( (i>=0) && (j>=0) && (i<H) && (j<W) )
	// 	d_patchCube[k] = d_image[ i*W + j ];
	// else
	// 	d_patchCube[k] = 0; // this is the zero-padding occuring in array boundaries...
}

__device__
float gaussian2D(float x, float y, float s_x, float s_y)
{
	float a = x*x/(s_x*s_x);
	float b = y*y/(s_y*s_y);

	return exp( -(a+b)/2 );
}



__global__
void apply_gaussian_filter(float *d_patchCube, int pH, int pW, float patchSigma_h, float patchSigma_w)
{	
	// Local coordinates on patch window (ref pixel at the center)
	int patch_i = threadIdx.y - (pH-1)/2;
	int patch_j = threadIdx.x - (pW-1)/2;

	// Calc Gaussian Filter value on patch coordinates
	float gaussCoeff = gaussian2D(patch_i, patch_j, patchSigma_h, patchSigma_w);

	// Multiply by existing patch-element
	int k = getGlobalIdx_2D_2D();
	d_patchCube[k] = gaussCoeff*d_patchCube[k];
}


__global__
void calc_dist_matrix(float *d_distMatrix, float *d_patchCube, int N, int M, float sigma)
{
	
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	// For efficient use of the cuda kernels, we should check for oversubscribtion 
	if( (i >= N) || (j>=N) )
		return;
	
	// Calc Dist	
	float D = 0;
	float a,b;
	for(int m=0; m < M; m++)
	{
		a = d_patchCube[i*M+m];
		b = d_patchCube[j*M+m];
		D += (a-b)*(a-b);
	}

	d_distMatrix[i*N+j] = exp(-D/sigma/sigma); 
}

__global__
void clip_dist_diag(float* d_dist, float* d_diag, int N)
{
	// int tid = blockIdx.x*gridDim.x + threadIdx.x;
	int tid = getGlobalIdx_2D_2D();
	
	// Incase we oversubscribe threads/blocks
	if(tid < N)
	{
		if(d_diag ==0)
			d_dist[tid*N+tid] = 0;
		else
		{
			float d = d_diag[tid];
			d_dist[tid*N+tid] = ( (d > FLT_MIN) ?  d:FLT_MIN) ;
		}	
	}
}

__global__
void multi_mat_vector_row(float* d_A, float* d_x, /*out*/ float* d_B, int N)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;

	// For efficient use of the cuda kernels, we should check for oversubscribtion 
	if( (i>=N) || (j>=N) )
		return;

	d_B[i*N+j] = d_A[i*N+j]*d_x[j]; 
	// d_B[i*N+j] = 0; //d_x[j]; 

}



__global__
void div_vector(float* d_A, float* d_x, /*out*/ float* d_B, int N)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	// For efficient use of the cuda kernels, we should check for oversubscribtion 
	if(i >= N)
		return;

	if(d_x[i] >  FLT_MIN)
		d_B[i] = d_A[i]/d_x[i]; 
	else
		d_B[i] = d_A[i]/FLT_MIN; 
}