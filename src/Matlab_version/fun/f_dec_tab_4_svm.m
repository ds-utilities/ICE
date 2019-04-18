function [err_mat,pred_prob_mat, y_pred_whole,auc_whole]=...
    f_dec_tab_4_svm(X, y, clus, useParfor, op)
% This version changed from the cluster-cluster err_mat to instance-cluster
%  err_mat. This solution will avoid the case that if one cluster decision 
%  is wrong leading entrie cluster prediction is wrong, which is the reason
%  of instability. However, we cannot use a systematic evaluation criteria 
%  such as AUC, I will try using the predicted prob at first. 

% This version 3 adds the support for fuzzy clustering - one instance may
%  belongs to more than one cluster. 
% This updated version also outputs the predicted values of y.
% support more than 3 clusters
% normalization take place in y_pred_self and y_pred_other, thus do not
%  need normalization when predict y_pred_tactic.
% ixsp is another cluster form.

if nargin < 5
    %adv_whole = 0.05;
    %adv_self = 0.03;
    %adv_whole = 0.02;
    %adv_self = 0.01;
    op = ' ';
end

n_clusters = length(clus);
pred_prob_mat = nan(length(y), n_clusters + 1);

% err_mat stores the prediction error.

fo_inner = 10;

% ixsp(ixsp == 0) = nan;
%% --------------------------- WHOLE -------------------------------------
% Predict each cluster using the whole data. 
% if useParfor == 1
[y_pred_whole, auc_whole] = f_weka_svm_arff_k_fo_3_parfor(X, y, fo_inner,op);

%[y_pred_whole, auc_whole] = f_weka_RF_arff_k_fo_3(X, y, fo_inner);

pred_prob_mat(:, n_clusters+1) = y_pred_whole;

fprintf('done whole evaluation\n');
%% ------------------------ SELF --------------------------------
% predict the whole instances using each cluster data, while self 
%  prediction using 10-fold CV.
[y_pred_multi]=f_use_each_clus_forWhole_svm(X, y, clus, y_pred_whole, useParfor,op);
pred_prob_mat(:, 1:n_clusters) = y_pred_multi;


err_mat = abs(pred_prob_mat - repmat(y, 1, n_clusters+1));





end










