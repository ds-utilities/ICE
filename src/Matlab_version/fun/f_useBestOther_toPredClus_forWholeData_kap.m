function [dec_mat_kap, y_pred_other, kap_other ] = ...
    f_useBestOther_toPredClus_forWholeData_kap(X, y, clus, dec_mat_kap, n_clusters, ...
    y_pred_whole, y_pred_self_multi )
% 
% use the best 'other' to predict current cluster for the whole data

y_pred_other = nan(length(y), 1);

% try
for c=1:n_clusters % c: for each col - each testing cluster
    %c,
    %dec_1_clust = dec_mat(1:end-1, c);
    dec_1_clust_kap = dec_mat_kap(1:end-1, c);
    
    % 'remove' self
    % dec_1_clust(c) = -inf;
    dec_1_clust_kap(c) = -inf;
    % find the best 'other' cluster to predict the current cluster.
    r = find(dec_1_clust_kap == max(dec_1_clust_kap));
    % if there are multiple best, just randomly choose one. 
    r = r(randi(length(r))); 
    % cluster 'r' is the best 'other' cluster to be the training data
    
    X_te = X(clus{c}, :);
    y_te = y(clus{c});    
    X_tr = X(clus{r}, :);
    %clus{r},
    y_tr = y(clus{r});
    [y_pred] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
    
    % Normalization
    templete = y_pred_whole(clus{c});
    target = y_pred;
    y_pred = f_my_quantileNorm(templete, target);

    % change the biased predicted instances (the ones also belongs to
    % the training insts.) to un-biased predictions. Change these to
    % clus{r}'s prediction
    tf_te = false(length(y), 1);
    tf_te(clus{c}) = true;
    tf_tr = false(length(y), 1);
    tf_tr(clus{r}) = true;
    % insts. for both cluster:
    tf_both = tf_te & tf_tr;
    y_pred_corrected = y_pred_self_multi(tf_both, r);
    tf_biased_on_te = tf_both(tf_te);

    y_pred(tf_biased_on_te) = y_pred_corrected;

        
    % put the normed prediction to the whole data.
    y_pred_other(clus{c}) = y_pred;
end
% catch exception
% end

y_tmp = y(~isnan(y_pred_other)); 
y_pred_other_noNan = y_pred_other(~isnan(y_pred_other));
% auc_other = f_SampleError(y_pred_other_noNan, y_tmp, 'AUC');
kap_other = f_my_01_kappa(y_pred_other_noNan, y_tmp);

fprintf('done using best other clusters to predict each cluster for whole data\n');

y_pred_other(isnan(y_pred_other)) = nanmean(y_pred_other);






end

