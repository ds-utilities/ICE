function [y_pred, auc, kap] = f_weka_RF_arff_k_fo_3(X, y, k_fold, I)
% This version is a revision on f_weka_RF_arff_k_fo(), that it won't use
%  the weka -x option for the cross-validation, rather, it uses train and
%  test basic weka commond. This is because of that later we will use the
%  predicted label prob and select a certain cluster of insts. But weka's
%  build-in 10 fold won't record the correct instances order. 

% k_fold = length(y);
if nargin < 3
    k_fold = 10;
    I = 100;
end


y_pred = zeros(size(y));
[train, test] = f_Kfold_cv(y, k_fold);
for i=1:k_fold
    X_te = X(test{i}, :);
    y_te = y(test{i});
    
    X_tr = X(train{i}, :);
    y_tr = y(train{i});
    
    y_pred(test{i}, :) = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te, I);
end

auc = f_SampleError(y_pred, y, 'AUC');
% kap = f_my_01_kappa(y_pred, y);
kap = NaN;

end






