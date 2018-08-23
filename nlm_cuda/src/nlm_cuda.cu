


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
	
	// Local coordinates on patch window (ref pixel at the center)
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

__device__
float gaussian2D(float x, float y, float s_x, float s_y)
{
	float a = x*x/(s_x*s_x);
	float b = y*y/(s_y*s_y);

	return expf( -(a+b)/2 );
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
void calc_dist_matrix(float *d_distMatrix, float *d_patchCube, float sigma)
{
	int i = blockIdx.y*gridDim.y + blockIdx.y;
	int j = blockIdx.x*gridDim.x + blockIdx.X;


	d_distMatrix[k] = d_patchCube

}