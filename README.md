# Non-Local Means Algorithm in CUDA

- The data folder is where the input and output data of the nlm-cuda implementation are stored. 
- The matlab folder contains useful functions for parsing .bin files in the data folder or converting .mat to .bin files, etc
- the opencv_performance_test folder contains a test-program, measuring the time performance of the opencv::fastNlMeansDenoising function

## Compile NLM-CUDA:
$ cd nlm_cuda
$ make all

## Run Demo:
$ ./demo_nlm_cuda.out ../data/house50.bin.in

## Run Benchmarks:
$ ./nlm_benvhmarks.out ../data/

## Run Unit Testing
$ ./test_bed 100

## Compile and Run opencv_performance_test/fastNlMeans
$ cd opencv_performance_test/fastNlMeansDenoising_Performance/
$ cmake .
$ make
$ ././openCV_FastNLM_test house.bmp