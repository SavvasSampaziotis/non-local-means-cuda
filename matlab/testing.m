
N = 10;
M = 5;
sigma = 10;


A = reshape(1:(N*M), M, N)'
D = squareform( pdist(A, 'euclidean'));

d = exp(-D.^2 /sigma/sigma)