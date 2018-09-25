
clear

Xo = parse_data_bin('../data/house50.bin.in');
X = parse_data_bin('../data/filtered.bin.out');
Xh = parse_data_bin('../data/filtered_adapt_h.bin.out');
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

figure(1); clf;
data = {Xo,X,h,Xh}
name = {'Original', 'Filtered: h_i_j=0.1','h_i_j per pixel', 'Filtered: h_i_j variable' };

for i=1:4
   subplot(2,2,i);
   imshow(data{i});
   title(name{i});
end