function auc = f_eval_simple_LR(X,y, k_fold, model_type)
%
[train_sets, test_sets] = f_StratifiedKFold(length(y), k_fold);
auc_folds = zeros(length(test_sets), 1);
% parfor fold = 1:length(test_sets)
for fold = 1:length(test_sets)
    % ------------------------ prepare data -----------------------------
    fold,
    train_loc = train_sets{fold, 1};
    test_loc = test_sets{fold, 1};
    train = f_loc_to_tf(train_loc, length(y));
    test = f_loc_to_tf(test_loc, length(y));

    X_train = X(train, :);
    y_train = y(train, :);
    X_test = X(test, :);
    y_test = y(test, :);
    
    % ------------------------ Evaluation -----------------------------
    [Y_predicteds] = f_main_simple(X_train,y_train, ...
        X_test, model_type);
    
    % ------------------------ Compute AUC -----------------------------
    auc = f_SampleError(Y_predicteds, y_test, 'AUC');
    auc_folds(fold, 1) = auc;
    
end
auc = mean(auc_folds);

end

