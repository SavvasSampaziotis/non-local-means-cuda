clear
clc
DIR = "../diades/";

mse = @(A,B) mean((A(:)-B(:)).^2);

hist2 = @(A,B) hist([A(:),B(:)]);

%% Patch Cube Gauss
P = parse_data_bin(DIR+'patchCube.bin');
H = fspecial('gaussian',[5,5], 5/3);
H = H(:) ./ max(H(:));
B2 = bsxfun( @times, P, H' );

Pg = parse_data_bin(DIR+'patchCube_gaussed.bin');
mse(B2,Pg)
hist2(B2,Pg)

%% Dist Matrix
D = squareform( pdist( B2, 'euclidean' ) );
D = exp( -D.^2 / 0.1^2 );

Dd = parse_data_bin(DIR+'dist.bin');
mse(D,Dd)
hist2(D,Dd)
surf(Dd-D)

%% Find max

dia = max(D-diag(diag(D)),[],2);
dia_d =  parse_data_bin(DIR+'max_diag.bin');
mse(dia,dia_d)
hist2(dia,dia_d)
plot([dia,dia_d])

%% Dist Diag Clip
D(1:length(D)+1:end) = max( max(D-diag(diag(D)),[],2), eps);

Dd2 = parse_data_bin(DIR+'distClipped.bin');

mse(D,Dd2)
hist2(D,Dd2)

%% Row Sum

Dsum = sum(D,2);

Dsumd = parse_data_bin(DIR+'dist_rowsum.bin');

mse(Dsum,Dsumd)
hist2(Dsum,Dsumd)


%% dist*image multiplication
load('house.mat');
I = house(1:8:end,1:8:end);
DI = D*I(:);

DId = parse_data_bin(DIR+'dist_image_mult.bin');

mse(DI,DId)
hist2(DI,DId)

%% Image Out

Y = parse_data_bin(DIR+'filtered_image.bin');
imshow(Y)
