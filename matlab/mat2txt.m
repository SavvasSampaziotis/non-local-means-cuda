
clear

load('house.mat')
Im = house;
filename = 'house';

% filename = 'test';
% Im = reshape( 1:16, 4, 4)


DATA_DIR = '../data/';
fid = fopen(strcat(DATA_DIR , filename ,'.dat'), 'w');
fwrite(fid, size(Im), 'int' );
fwrite(fid, Im, 'float');
fclose(fid);

% pixels are saved in float precision, because my machine runs CUDA 1.1, 
% supporting only single precision floating point arithmetic
