


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


__global__
void generate_3D_cube(float *d_image, float *d_patchCube, int H, int W, int pH, int pW)
{
	int k = getGlobalIdx_2D_2D();	
	
	// Translate the z-coordinate to local patch coordinates
	int patch_i = threadIdx.y - (pH-1)/2;
	int patch_j = threadIdx.x - (pW-1)/2;
	
	int im_i = blockIdx.y;
	int im_j = blockIdx.x;

	int i = im_i+patch_i;
	int j = im_j+patch_j;

	if ( (i>=0) && (j>=0) && (i<H) && (j<W) )
		d_patchCube[k] = d_image[ i*W + j ];
	else
		d_patchCube[k] = 0; // this is the zero-padding occuring in array boundaries...

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