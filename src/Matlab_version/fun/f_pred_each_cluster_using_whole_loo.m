function [dec_mat, y_pred_whole, auc_whole] = ...
    f_pred_each_cluster_using_whole_loo(X, y, clus, dec_mat, n_clusters, ...
    useParfor)
% Predict each cluster using the whole data.


% try
% [y_pred_whole, auc_whole] = f_weka_RF_arff_10_fo_3(X, y);
if useParfor == 1
[y_pred_whole, auc_whole] = f_weka_RF_arff_k_fo_3_parfor(X, y, length(y));
else
[y_pred_whole, auc_whole] = f_weka_RF_arff_k_fo_3(X, y, length(y));
end

for j=1:n_clusters        
    auc = f_SampleError(y_pred_whole(clus{j}), y(clus{j}), 'AUC');
    dec_mat(n_clusters+1, j) = auc;
end

fprintf('done using whole data to predict each clusters\n');
% catch exception   
% end



end

