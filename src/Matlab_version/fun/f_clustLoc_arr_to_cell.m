function c = f_clustLoc_arr_to_cell(arr)
% convert the cluster indicator format from array to cell

c={};
u = unique(arr);
for i=1:length(u)
    c{i} = find(arr == u(i));    
end

end