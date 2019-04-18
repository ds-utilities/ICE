function [ixs] = f_find_centers_rwMat(rw_mat, k)
% on the rw_mat matrix, find some nodes as the centers for soft
%  clustering. If we just random pickup some nodes as centroids, that is
%  not good for fuzzy clusters. 
% k is the number of centroids. 

ixs = [];
% 1. find the most connected center node as the first centroid. 
a = sum(rw_mat, 2);
% most connected node.
ix = find(a == max(a));
ixs = [ixs, ix(1)];

% 2. iteratively find the rest nodes
for i=2:k
    b = sum( rw_mat(:, ixs), 2);
    b(ixs) = inf;
    
    % find the farthest node
    ix = find(b == min(b));
    ix = ix(1);
    ixs = [ixs, ix];
end





end

