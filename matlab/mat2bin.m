
function [] = mat2bin(I, filename)

% clear
% load('house.mat')
% I = J;
% filename = 'house';

DATA_DIR = '../data/';
fid = fopen(strcat(DATA_DIR , filename ,'.bin.in'), 'w');
fwrite(fid, size(I), 'int' );
fwrite(fid, I, 'float');
fclose(fid);

% pixels are saved in float precision, because my machine runs CUDA 1.1, 
% supporting only single precision floating point arithmetic
end