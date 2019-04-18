function [nets] = f_rwMat_2_rwNets(rw_mat, neighb_ver, low_per, high_per)
% Generate RW nets based on RW matrix
% low_per:  neighborhood percentage at least to the whole instances.
% high_per: neighborhood percentage at  most to the whole instances.


% determine the neighborhood sizes.

if neighb_ver == 2
    szs = f_gen_neighbor_count_cutoff_2(length(rw_mat));
elseif neighb_ver==3
    szs = f_gen_neighbor_count_cutoff_3(length(rw_mat), low_per, high_per);
end

cutoffs = getCutoffs(rw_mat, szs);
min_neighbors = min(szs);

nets = cell(1, length(szs));
for i=1:length(cutoffs)
    rw_net = rw_mat > cutoffs(i);
    rw_net = f_fill_rand_neighbors(rw_net, min_neighbors);
    
    nets{1, i} = logical(rw_net);
    %nets{1, i} = sparse(rw_net);
end


end

