
clear

Xo = parse_data_bin('../data/house32.bin.in');
X = parse_data_bin('../nlm_cuda/imout.bin.out');
Xh = parse_data_bin('../nlm_cuda/imout_h.bin.out');
h = zeros(size(Xo));
[H,W] = size(h);
for i=1:H
    for j=1:W
        if i<(0.75*H) && i>(H/4) && j<(0.75*W) && j>(W/4) 
            h(i,j) = 1;
        else
            h(i,j)= 0.1;
        end
    end
end

% figure(1); clf;
data = {Xo,X,h,Xh}
name = {'Original', 'Filtered: h_i_j=0.1','h_i_j per pixel', 'Filtered: h_i_j variable' };
dir = '~/Desktop/WinShare/figs/';

for i=1:4
   imshow(data{i});
   title(name{i});
   savefig(strcat(dir,name{i},'.fig'))
end