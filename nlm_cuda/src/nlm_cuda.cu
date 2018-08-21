


/*Source: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf */
__device__
int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}


__global__
void generate_3D_cube(float *d_image, float *d_patchCube, int patchSize_W, int patchSize_H)
{
	
}


__global__
void apply_gaussian_filter(float *d_image, float *d_patchCube, int patchSize, int )
{
	// generate gauusian

	// multiply by patch 

	// normalize my 
}


__global__
void calc_dist_matrix(float *d_distMatrix, float *d_patchCube, float sigma)
{
	
}