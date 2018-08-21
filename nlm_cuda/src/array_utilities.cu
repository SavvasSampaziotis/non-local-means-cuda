#include <stdio.h>
#include <stdlib.h>


void read_dataset(int* H, int* W, float** data, const char* filename)
{
	FILE *fp = fopen(filename, "r");
	if(fp < 0)
	{
		printf("ERROR OPENING DATA FILE\n");
		*H = 0;
		*W = 0;
		return;
	}

	fread(H, 1, sizeof(int), fp);
	fread(W, 1, sizeof(int), fp);
	
	int L = (*H)*(*W);
	*data = (float*) malloc( L*sizeof(float));
	fread(*data, L, sizeof(float), fp);

	printf("\n[READ_IMAGEDATA]: %s\tH=%d\tW=%d Read Succesfully\n",filename, *H,*W);
	
	fclose(fp);
}