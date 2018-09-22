/*
	Source File for demo nlm executable

	Author: Savvas Sampaziotis
*/


#include <stdio.h>
#include <stdlib.h>

#include "array_utilities.h"
#include "nlm_cuda.h"
// #include "time_measure.h"

int main(int argc, char** argv)
{
	// TimeInterval totalTime;

	int patch_H = 5;
	int patch_W = 5;
	float patchSigmaH = 1.66;
	float patchSigmaW = 1.66;
	float sigma = 0.1f;

	char* filename;
	if(argc == 2)
	{
		printf("file chosen: %s\n", argv[1]);
		filename = argv[1];
	}
	else
	{
		printf("PLEASE GIVE FILE NAME AS PROGRAM INPUT\n");
		return 0;
	}

	/*Local Host Pointers*/
	float *image_in; // H-by-W  Image, stored in continues 1-by-N array (N = H*W)
	float *image_out; //output image
	int H,W;
	
	/* Read Data */
	read_dataset(&H, &W, &image_in, filename);
	image_out = (float*) malloc(H*W*sizeof(float));

	// tic(&totalTime);
	
	double calcTime =nonLocalMeansCUDA(image_in, image_out, H,W, patch_H, patch_W, patchSigmaH, patchSigmaW, sigma);

	// toc(&totalTime);
	// printf("%f\n", totalTime.seqTime);
	printf("%f\n", calcTime);
		
	write_datfile(H,W,image_out, "./imout.bin");

	free(image_in);
	free(image_out);
}