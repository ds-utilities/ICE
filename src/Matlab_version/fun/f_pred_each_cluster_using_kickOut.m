function [dec_mat, y_pred_kickOut_whole] = ...
    f_pred_each_cluster_using_kickOut(X, y, clus, clus_kickOut, dec_mat, n_clusters)
% Predict each cluster using the whole data kicked out one cluster.

n_r = size(dec_mat, 1);

tf = false(length(y), 1);
tf(clus_kickOut) = true; % only the new cluster shows 1.
X_rest = X(tf, :);
y_rest = y(tf);

X_kickedOut = X(~tf, :);
y_kickedOut = y(~tf);

y_pred_kickOut_whole = zeros(length(y), 1);

try
% for those in the data, using 10-fold:
[y_pred_tmp, auc] = f_weka_RF_arff_10_fo_3(X_rest, y_rest);
y_pred_kickOut_whole(tf) = y_pred_tmp;

% for those kicked out, using train-test:
y_pred = f_weka_RF_tr_te(X_rest, y_rest, X_kickedOut, y_kickedOut);
y_pred_kickOut_whole(~tf) = y_pred;



for j=1:n_clusters        
    auc = f_SampleError(y_pred_kickOut_whole(clus{j}), y(clus{j}), 'AUC');
    %dec_mat(n_clusters+1, j) = auc;
    dec_mat(n_r+1, j) = auc;
end

fprintf('done using kick-out-whole data to predict each clusters\n');
catch exception   
end



end

