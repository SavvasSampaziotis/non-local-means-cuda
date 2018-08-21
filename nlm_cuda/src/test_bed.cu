

#include "array_utilities.h"
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv)
{
	
	char* filename;
	if(argc == 2)
	{
		filename = argv[1];
	}

	printf("[MAIN]:\t.dat File targeted for denoising: %s", filename);

	int H,W;
	float *data;
	read_dataset(&H, &W, &data, filename);

	printf("%f\n\n",data[0]);




	

	
	return 0;
}
