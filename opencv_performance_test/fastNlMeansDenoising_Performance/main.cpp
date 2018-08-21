#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <opencv2/photo.hpp>
#include <opencv2/photo/cuda.hpp>


#include "time_measure.h"

using namespace cv;
int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    // waitKey(0);


    Mat image2;

    TimeInterval ti;
    tic(&ti);
    fastNlMeansDenoising( image, image2, 3, 7, 21);
    double seqTime = toc(&ti);
    printf("Time Elapsed: %f\n", seqTime);

    // cv::cuda::nonLocalMeans(image, image3, 3,21,7,BORDER_REFLECT101);

    if ( !image2.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image2);
    waitKey(0);

    // imwrite(image2, '')

    return 0;
}