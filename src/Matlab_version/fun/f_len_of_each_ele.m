function lens = f_len_of_each_ele(c1)
% Assume c1 is a 1-dimension cell array, and each element is a 1d double
%  array. This function counts the length of each double array. 

lens = zeros(size(c1));
for i=1:length(c1)
    lens(i) = length(c1{i});
end


end

