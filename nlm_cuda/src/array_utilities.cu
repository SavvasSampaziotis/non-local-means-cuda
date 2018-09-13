#include <stdio.h>
#include <stdlib.h>


void write_datfile(int N, int D, float* data, const char* name)
{
	FILE *fp = fopen( name, "w+");
	
	if(fp < 0)
	{
		printf("ERROR OPENING DATA FILE\n");
		return;
	}

	fwrite(&N, 1, sizeof(int), fp);
	fwrite(&D, 1, sizeof(int), fp);
	fwrite(data, D*N, sizeof(float), fp);

	fclose(fp);
}


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

	// printf("\n[READ_IMAGEDATA]: %s\tH=%d\tW=%d Read Succesfully\n",filename, *H,*W);
	
	fclose(fp);
}





void print_array(int H, int W, float* data)
{
	int i,j;
	// printf("\n--------------------\n");
	for(i=0; i<H; i++)
	{
		for(j=0; j<W; j++)
		{
			 printf("%f ", data[i*W+j]);
		}
		printf("\n");
	}
		

}



