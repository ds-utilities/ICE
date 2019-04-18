function [clus] = get_fuzzy_rwr_clus_mt_sz(rw_mat, each_clus_szs , k,ixs_centers)
% get all the clusters with multi size
clus = {};
for i = 1:length(each_clus_szs)
    each_clus_sz = each_clus_szs(i);
    cls = get_fuzzy_rwr_clus(rw_mat, each_clus_sz, k,ixs_centers );
    
    clus = [clus, cls];
    
    i,
end

end