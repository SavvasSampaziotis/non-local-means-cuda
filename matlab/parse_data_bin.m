function X =parse_data_bin(filename)

fileID = fopen(filename, 'r+');
if fileID < 0
    disp('ERROR')
end
H = fread(fileID, 1, 'int32');
W = fread(fileID, 1, 'int32');
X = fread(fileID, H*W, 'float'); % Data are written in column order...
fclose(fileID);

X = reshape(X,H,W);

end