
clear;

fileID = fopen('../data/out.bin', 'r+');


H = fread(fileID, 1, 'int32');
W = fread(fileID, 1, 'int32');
X = fread(fileID, H*W, 'float'); % Data are written in column order...
fclose(fileID);

X = reshape(X,H,W);

image(X)