function [ auc, y_pred ] = f_weka_bg_svm_kfo( X, y , op )
% 
I = 100;
if nargin < 3
    op = ['  -I ',int2str(I), '  -W weka.classifiers.functions.SMO  '];
end
k_fold = 10;

y_pred = zeros(length(y));
[train, test] = f_Kfold_cv(y, k_fold);

for k=1:k_fold
    k,
    % train and test index
    X_te = X(test{k}, :);
    y_te = y(test{k});
    X_tr = X(train{k}, :);
    y_tr = y(train{k});

    %[clus] = f_fuzzy_rwr_clusters(X_tr, n_clus);
    %tfs = f_clus_to_tfs(clus, size(X_tr, 1) );
    % y predictions using adaboost  

    y_pred = f_weka_bg_svm_tr_te(X_tr, y_tr, X_te, y_te, op); 
    y_pred(test{k}, 1) = y_pred;

end
auc = f_SampleError(y_pred, y, 'AUC');

end

