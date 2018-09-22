load('house.mat')


S = [8, 10, 16, 32, 64];
for i=1:length(S)
    I = house(1:S(i),1:S(i));
    filename = strcat('house',num2str(S(i)))
    mat2bin(I, filename);
end


