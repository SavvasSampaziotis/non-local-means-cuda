/*
	Source File for row-wise reduction tools library

	Author: Savvas Sampaziotis
*/

#include "reduction.h"

void row_sum_WR(int N, float* d_X, /*out*/ ReductionCache* rc)
{
	reduction_invoker(N,d_X,rc, R_SUM);
}

void row_max_WR(int N, float* d_X, /*out*/ ReductionCache* rc)
{
	reduction_invoker(N,d_X,rc, R_MAX);
}

void reduction_invoker(int N, float* d_A, /*out*/ ReductionCache* rc, int op )
{
	if(rc->blocksNum == 1)
	{
		// We need only one reduction call!
		reduction <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, d_A, rc->d_sum, op);

		//no need for the d_reduction cache 			
	}
	else // We need multiple reduction calls!
	{	
		/* Level 0 Reduction - Multiple Blocks */
		reduction <<<rc->gridDim, rc->blockDim, rc->cache_size>>>(N, d_A, rc->d_reduced_vec, op);		
			
		/* Level 1 Reduction - Single Block */

		// Ideally we would like threads_num==length(reduced_vec)/numRow. However threads_num2 must be a power of 2. Thus:
		int threads_num2 = exp2f(floor(log2f(rc->reduced_vec_length/rc->rowNum))); 
		if(threads_num2>512)
			threads_num2=512;
		
		//printf("[REDUCTION WRAPPER:] Level 0: THREADS: %d RED_VEC %d\n", threads_num2, rc->blocksNum );
		
		dim3 gridDim2(1,rc->rowNum,1);
		dim3 blockDim2(threads_num2,1,1);
		reduction <<<gridDim2, blockDim2, threads_num2*sizeof(float)>>>\
			(rc->gridDim.x, rc->d_reduced_vec, rc->d_sum, op); //

		// WARNING: launching with original thread_num might be too much. 
		// SOLUTION: Find power-of-2 nearest to block_num 
	}	
}

__device__
float reduction_op_init(int op)
{
	switch(op)
	{
		case R_SUM:
			return 0;
		case R_MAX:
			return -FLT_MAX;
		default:
			return 0;
	}	
}

__device__
float reduction_op(float a, float b, int op)
{
	switch(op)
	{
		case R_SUM:
			return a+b;
		case R_MAX:
			return (a>b) ? a:b;
		default:
			return a+b;
	}
}


/*
	This is the reduction function used for a variety of matrix operations. 

*/
__global__
void reduction(int N, float* X, float* reducted_vec, int op)
{
	extern __shared__ float reduction_cache[] ;

	//thread ID on each row of blocks
	int tid = blockDim.x * blockIdx.x + threadIdx.x; 
	int cache_i = threadIdx.x;

	int tid_y = blockIdx.y;

	/* This UNROLLS the elements of x, "outside" the grid's index range.
		In the case of N=600, threadsPerBlock=256 and 2 blocks in total, 
		we have 600-256*2=88 additions done in parallel, before the reduction of the 512 threads.

		incase the index-range > N, the reduction scheme will simply add some zeros to the vector. 
		This allows as to oversubscribe in terms of threads and blocks. 
	*/
	int offset = N*tid_y;
	float temp = reduction_op_init(op); // 0 or MIN_FLOAT
	while (tid < N)
	{
		temp = reduction_op(temp, X[tid+offset], op); 
		tid += blockDim.x * gridDim.x;
	}

	/* Load x-data  into local shared memory. 
		As mentioned before, some entries are small sums of
		 x's outside the grid's range  */
	reduction_cache[cache_i] = temp;	
	__syncthreads();
	
	// Begin the reduction per shared-memory-block
	for(unsigned int s = blockDim.x/2; s > 0; s>>=1)
	{	
		if(cache_i < s)
			reduction_cache[cache_i] = reduction_op(reduction_cache[cache_i], reduction_cache[cache_i+s], op);  
		__syncthreads();
	}

	// Unroll Last warp
	/*if(cache_i>32)
	{
		reduction_cache[cache_i] += reduction_cache[cache_i+32];
		reduction_cache[cache_i] += reduction_cache[cache_i+16];
		reduction_cache[cache_i] += reduction_cache[cache_i+8];
		reduction_cache[cache_i] += reduction_cache[cache_i+4];
		reduction_cache[cache_i] += reduction_cache[cache_i+2];
		reduction_cache[cache_i] += reduction_cache[cache_i+1];  
	}*/

	// Final Sum is stored in global array.
	if(cache_i==0)
		reducted_vec[blockIdx.y*gridDim.x + blockIdx.x] = reduction_cache[0];
}



/*

*/
void init_reduction_cache(int rowLength, int rowNum, int threads_num, /*out*/ ReductionCache* rc)
{
	rc->blockDim.x = threads_num;
	rc->blockDim.y = 1;
	rc->blockDim.z = 1;

	int blocks_num = ceil(rowLength/threads_num); 
	if(blocks_num==0) blocks_num=1;

	rc->blocksNum = blocks_num;

	rc->gridDim.x = blocks_num;
	rc->gridDim.y = rowNum; // One row of block for each matrix row -> we need the extra space for the Shared Memory anyway... 
	rc->gridDim.z = 1;

	rc->rowNum = rowNum;
	rc->reduced_vec_length = rowNum*blocks_num; // ronNum * (number of blocks per row) 
	
	rc->cache_size = threads_num*sizeof(float);
	if(rc->cache_size > 1024*16) // cache > 16 KB. CUCA 1.x allows max sm 16MB per Multi-Processor
	{
		printf("[WARNING]:\t[INIT_REDUCTION_CACHE]:\tShared Memory size too large: %lu\n", rc->cache_size);
		// rc->cache_size = 
	}

	if(blocks_num>1)  
		cudaMalloc((void**) &(rc->d_reduced_vec), rc->reduced_vec_length*sizeof(float));
		// This is not needed in this case. As reduction cache, d_sum can also be used.

	cudaMalloc((void**) &(rc->d_sum), rowNum*sizeof(float));
}


void delete_reduction_cache(ReductionCache* reductionCache)
{
	if(reductionCache->blocksNum>1)
		cudaFree(reductionCache->d_reduced_vec);
	cudaFree(reductionCache->d_sum);
}

