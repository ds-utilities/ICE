function clus = f_fuzzy_rwr_clusters_v3(X, k , each_clus_szs)
% Version 2, the clusters size is more adaptive.
% Version 2, each center point has multi cluster size. 
% The fit-cluster size of previous setting is a downside. 

% X: data
% k: number of clusters

nSteps = 1000;
lazi = 0.3;

% clus = {};
%%
sim = squareform(pdist(X)); 
sim = -sim;

% if do_rwr
% ----------------------------------------
% fprintf('    done RWR\n');
% figure,
% imagesc(sim);
% ----------------------------------------

aRank_k_neighbors = ceil(log10(length(sim)) ) ;
ori_graph = f_sim_2_aRankNet(sim, aRank_k_neighbors);

% -------- RWR --------
% nSteps = 1000;
% lazi = 0.3;


rw = RWR(ori_graph, nSteps, lazi);
% remove probability of returning start node
rw_mat = rw-diag(diag(rw));
% ----------------------------------------
% fprintf('    done RWR\n');
% figure,
% imagesc(rw_mat);
% ----------------------------------------
% else
%     rw_mat = sim;
% end


%%
ixs_centers = f_find_centers_rwMat(rw_mat, k);
ixs_centers,

% overlap_factor = 2;
% overlap_sz = 1.7 .* size(X, 1);
% ---------------------------------------
% overlap_sz = overlap_factor .* size(X, 1);
% ---------------------------------------
% overlap_sz = 1.8 .* size(X, 1);

[clus] = get_fuzzy_rwr_clus_mt_sz(rw_mat, each_clus_szs, k,ixs_centers );

%% sort the clusters
[sorted_sz, ix] = sort(f_len_of_each_ele(clus), 'descend');
clus = clus(ix);
fprintf('cluster size: ');
sorted_sz,
fprintf('\n');


end

% function cls = get_fuzzy_rwr_clus(rw_mat, each_clus_sz,k,ixs_centers)
% % for one cluster size, get the cluster
% cutoffs = getCutoffs(rw_mat, each_clus_sz);
% ct = cutoffs(1);
% rw_net = rw_mat > ct;
% 
% % set the diagnal to 1
% rw_net(1:length(rw_net)+1: numel(rw_net)) = 1;
% 
% for i=1:k
%     %clus{1, i} = [find(rw_net(:, ixs_centers(i) ) ); ixs_centers(i) ];
%     cls{1, i} = find(rw_net(:, ixs_centers(i)) );
%     %length(clus{1, i}),
% end
% 
% end

% function [clus] = get_fuzzy_rwr_clus_mt_sz(rw_mat, each_clus_szs , k,ixs_centers)
% % get all the clusters with multi size
% clus = {};
% for i = 1:length(each_clus_szs)
%     each_clus_sz = each_clus_szs(i);
%     cls = get_fuzzy_rwr_clus(rw_mat, each_clus_sz, k,ixs_centers );
%     
%     clus = [clus, cls];
%     
% end
% 
% end





