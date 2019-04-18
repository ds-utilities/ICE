function [] = f_tactic_multi_clusters(X, y, n_clusters)
% 
simType = 'euclidean';

% Clustering
[clus, net] = f_do_clustering(X, 'kmeans', n_clusters, simType);
% sort the clusters
[sorted_sz, ix] = sort(f_len_of_each_ele(clus), 'descend');
clus = clus(ix);

% make a decision table and predict
% [dec_mat] = f_AB_dec_tab(X, y, clus);
[dec_mat, y_pred_whole, auc_whole, y_pred_self, auc_self, ...
    y_pred_other, auc_other, y_pred_tactic, auc_tactic] = ...
    f_AB_dec_tab_2(X, y, clus);



end

