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

int main(int argc, char** argv)
{

	int patch_H = 3;
	int patch_W = 3;
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

	timeval startwtime, endwtime;
	
	double calcTime;
	gettimeofday(&startwtime, NULL);

	calcTime = nonLocalMeansCUDA(image_in, image_out, H,W, patch_H, patch_W, patchSigmaH, patchSigmaW, sigma);

	double totalTime ;
	gettimeofday(&endwtime, NULL);

	totalTime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

	printf("Total Time:\t%fsec\n", totalTime);
	printf("Kernel Time:\t%fsec\n", calcTime);
		
	write_datfile(H,W,image_out, "./imout.bin.out");

	free(image_in);
	free(image_out);
}