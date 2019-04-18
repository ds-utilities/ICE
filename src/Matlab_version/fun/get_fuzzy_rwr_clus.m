function cls = get_fuzzy_rwr_clus(rw_mat, each_clus_sz,k,ixs_centers)
% for one cluster size, get the cluster
cutoffs = getCutoffs(rw_mat, each_clus_sz);
ct = cutoffs(1);
rw_net = rw_mat > ct;

% set the diagnal to 1
rw_net(1:length(rw_net)+1: numel(rw_net)) = 1;

for i=1:k
    %clus{1, i} = [find(rw_net(:, ixs_centers(i) ) ); ixs_centers(i) ];
    cls{1, i} = find(rw_net(:, ixs_centers(i)) );
    %length(clus{1, i}),
end

end