/*
	Source File for demo nlm executable

	Author: Savvas Sampaziotis
*/


#include <stdio.h>
#include <stdlib.h>

#include "array_utilities.h"
#include "nlm_cuda.h"
// #include "time_measure.h"
#include <sys/time.h>

/* NLM Params*/
float patchSigmaH = 1.66;
float patchSigmaW = 1.66;
float sigma = 0.1f;

/*Local Host Pointers*/
float *image_in; // H-by-W  Image, stored in continues 1-by-N array (N = H*W)
float *image_out; //output image
int H,W;

void sweepPatchSizes(const char* data_dir, const char* filename);

int main(int argc, char** argv)
{
	char* data_dir;
	if(argc == 2)
	{
		data_dir = argv[1];
	}
	else
	{
		printf("PLEASE PASS INPUT DATA DIR AS EXTRA ARG\n");
		return 0;
	}

	printf("\n");
	sweepPatchSizes(data_dir,"house10.bin.in");
	sweepPatchSizes(data_dir,"house16.bin.in");
	sweepPatchSizes(data_dir,"house32.bin.in");
	sweepPatchSizes(data_dir,"house64.bin.in");
	printf("\n");
}


void sweepPatchSizes(const char* data_dir, const char* filename)
{
	int RUNS = 20;
	char f[50];

	strcpy(f,data_dir);
	strcat(f,filename);
	
	// printf("%s\n", f);

	read_dataset(&H, &W, &image_in, f);
	image_out = (float*) malloc(H*W*sizeof(float));
	
	
	for(int patchSize=1; patchSize<=9; patchSize+=2)
	{
		double calcTime = 0;
		for (int i=0; i<RUNS; i++)
			calcTime += nonLocalMeansCUDA(image_in, image_out, H,W, patchSize, patchSize, patchSigmaH, patchSigmaW, sigma);
		calcTime = calcTime/RUNS;
		printf("%f, ", calcTime);
	}
	printf("\n");

	free(image_in);
	free(image_out);
}