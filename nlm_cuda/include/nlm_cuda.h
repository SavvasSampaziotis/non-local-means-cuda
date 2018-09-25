/*
	Header File for the non Local Means Imlementation Host Functions.

	Author: Savvas Sampaziotis
*/
#ifndef NLM_CUDA_H
#define  NLM_CUDA_H

#include "reduction.h"
#include "nlm_cuda_kernels.h"


/* 
	NLM Implementation for variable filter-paramater 'h'
	Main Caller: This function invokes the rest of the helper functions step-bu-step, thus realizing the complte NLM Algorithm.

	returns: The Calcultion Time EXCLUDING the data exchange between the host memory and the device memory 
*/
double nonLocalMeansCUDA_adapt(float* image_in, float *image_out, int H,int W, int patch_H, int patch_W, int patchSigmaH,int patchSigmaW, float* d_hMat);


/* 
NLM Implementation for constant and universal filter-paramater 'h'
	Main Caller: This function invokes the rest of the helper functions step-bu-step, thus realizing the complte NLM Algorithm

	returns: The Calcultion Time EXCLUDING the data exchange between the host memory and the device memory 
*/
double nonLocalMeansCUDA(float* image_in, float *image_out, int H,int W, int patch_H, int patch_W, int patchSigmaH,int patchSigmaW, float h);





#endif