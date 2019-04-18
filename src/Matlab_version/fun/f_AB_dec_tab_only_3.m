function [dec_mat] = f_AB_dec_tab_only_3(X, y, clus, useParfor)
% This version only compute for the dec_mat; it won't compute the fake
%  tactic.
% This version 3 adds the support for fuzzy clustering - one instance may
%  belongs to more than one cluster. 
% This updated version also outputs the predicted values of y.
% support more than 3 clusters
% normalization take place in y_pred_self and y_pred_other, thus do not
%  need normalization when predict y_pred_tactic.
% ixsp is another cluster form.


n_clusters = length(clus);
dec_mat = nan(n_clusters + 1, n_clusters);

% ixsp(ixsp == 0) = nan;
%% --------------------------- WHOLE -------------------------------------
% Predict each cluster using the whole data. 
[dec_mat, y_pred_whole, auc_whole] = ...
    f_pred_each_cluster_using_whole(X, y, clus, dec_mat, n_clusters,...
    useParfor);

%% ------------------------ SELF --------------------------------
% Self prediction using 10-fold CV
[dec_mat, y_pred_self, auc_self, y_pred_self_multi] = ...
    f_pred_each_cluster_using_self_forWhole(X, y, clus, dec_mat, ...
    n_clusters, y_pred_whole, useParfor);

%% ---------------------- Find best 'other' ------------------------------
% fill dec_mat for 'others'
[dec_mat ] = f_fill_dec_mat_for_others(X, y, clus, dec_mat, ...
    n_clusters, y_pred_whole, y_pred_self_multi);

%% use the best 'other' to predict current cluster for the whole data
%% tactically use the best for each cluster


end










