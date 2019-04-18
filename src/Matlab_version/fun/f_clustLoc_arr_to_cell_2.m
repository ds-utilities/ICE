function c = f_clustLoc_arr_to_cell_2(ixsp)
% convert the cluster indicator format from array to cell
% Updated from function f_clustLoc_arr_to_cell(). This version converts
%  clusters for fuzzy cluster result. 

% e.g. input:
%      2     3     2     2     5    ... 
%      0     2     3     0     0    ...
%      0     0     0     0     0    ...
%      0     0     0     0     0    ...
%      0     0     0     0     0    ...

c={};
u = unique(ixsp); % u: the cluster ids.
u = u(2:end); % delete 0, since here 0 is not a cluster id.  

for i=1:length(u) % for each cluster id
    inst_ids = [];
    for j=1:length(u) % for each row of the id array.
        inst_ids = [inst_ids, find(ixsp(j, :) == u(i)) ];  
    end
    c{i} = sort(inst_ids);  
end

end