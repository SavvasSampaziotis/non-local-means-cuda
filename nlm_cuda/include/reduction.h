
/*
	Header File for non LOcal Means Imlementation

	Author: Savvas Sampaziotis
*/
#ifndef REDUCTION_H
#define  REDUCTION_H


#define R_SUM 1
#define R_MAX 2

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/*
	This struct contains various metadata needed for reduction-kernel launching.
	
	It purpose is to make the main code more readable, and relieve me from numerous device-allocated arrays and blocksizes.

	This also enables me to write more efficient code, since memory (de)allocation needs to be kept at a minimum.

	Ref:
		https://www.cs.virginia.edu/~mwb7w/cuda_support/memory_management_overhead.html
*/
typedef struct ReductionCacheStruct
{
	
	dim3 gridDim;
	dim3 blockDim;
	float* d_reduced_vec;
	float* d_sum;
	size_t cache_size;
	int rowNum;
	int blocksNum; //==gridDim.x for sanity purposes mainly
	int reduced_vec_length; //for debugging purposes mainly
} ReductionCache;



/*
	Wrapper function for calling the base reduction Wrapper for the calc-sum of row reduction algorithm	
*/
void row_sum_WR(int N, float* d_X, /*out*/ ReductionCache* rc);

/*
	Wrapper function for calling the base reduction Wrapper for the find-max in row reduction algorithm
*/
void row_max_WR(int N, float* d_X, /*out*/ ReductionCache* rc);


/*
	launches 2 reduction-kernels: the level-0 and level-1 reduction scheme.
	The first one runs on gridDim and BlockDim specified by arg ReductionCache.
	The second one sums-up the reduction_vec produced by the first launch.
	The wrapper function is designed to handle row-wise operations on matrices    
*/
void reduction_invoker(int N, float* d_x,  /*out*/ ReductionCache* rc, int op);


/*
	"Constructor" of reductionCache struct.
	Input Args:
		int N: the length of the vector targeted for reduction
		int threads_num: Must be a power of 2. You need to choose its value wisely, 
	Output Args	
		ReductionCache* rc: the struct that holds the reduction metadata. 
*/
void init_reduction_cache(int N, int rowNum, int threads_num, /*out*/ ReductionCache* rc);


/*
	"Deconstructor" of reductionCache struct.
	Input Args:	
		ReductionCache* reductionCache

	This Basically deallocates the reduction cache vector from the GPU memory. 	
*/
void delete_reduction_cache(ReductionCache* reductionCache);



/*
	Simple add-reduction with SHARED MEMORY.
	This is the row-wise reduction function used for a variety of matrix operations
	
	Arguments
		int N:	length of x
		float* x
		float* reducted_vec[BLOCK_NUM]: reduced sum of each block
		int op: the reduction-operation in coded in integer. 1 for sum, 2 for max 
	 
	TIP:
		reduction_vec must be of size [BLOCK_NUM] and stores the reduced sum of each block of threads.
		
		The final result can be calculated by calling again reduction_SM for the reducted_vec like this (e.g):
		reduction_SM <<< 150/64, 64, cache_mem >>> (150, x, d_reduced_vec); // 3 blocks - 64 threads per block.
		reduction_SM <<< 1, 8, cache_mem >>> (N/256, d_reduced_vec, sum); // 1 block - 4 threads per block.
		 //There is ONE extra thread that will contribute zero to the final result. 

	Notes:
		Number of threads per block MUST be a power of 2. 

		Number of total threads blockNum*threadNum can be less or greater 
			than length of vector x. This works either way. 
			However, having less threads than vector elements mean that  there 
			will be more non-parallel global memory accesses. 

	Ref. Code
		http://cuda-programming.blogspot.gr/2013/01/vector-dot-product-in-cuda-c.html
		and
		Reduction.pdf from the courses material
*/
__global__
// void reduction(int N, float* X, float* reducted_vec, int tid_y, pfunc funct);
void reduction(int N, float* X, float* reducted_vec, int op);



#endif