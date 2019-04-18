function [y_preds, AUCs, stds]=f_read_y_pred_jamiul_java_out(path_res, ...
    n_insts, k_fold, n_cutoffs, y)
% 

[train_sets, test_sets] = f_StratifiedKFold(n_insts, k_fold);

y_preds = zeros(k_fold, n_cutoffs); % note that k_fold = n_insts

for fold = 1:k_fold
    
    % prepare y label for this fold
    fold,
    %train_loc = train_sets{fold, 1};
    test_loc = test_sets{fold, 1};
    %y_test_cur_fold = y(test_loc);
    
    % Read the y_predicted for all the cutoffs for the current fold
    fn = [path_res, 'na_fold_',int2str(fold),'.txt' ];
    [arr] = f_read_JamiulCode_Out(fn);
    
    if numel(arr) == 0
        y_pred_test_fold = nan(length(test_loc), n_cutoffs);
    else
        y_pred_test_fold = arr;
    end
    
    % test_loc,
    
    y_preds(test_loc, : ) = y_pred_test_fold;
    % size(arr)
    % length(y_test_cur_fold)
    
    % Compute AUCs for a fold:
end % fold


AUCs = zeros(1, n_cutoffs);
for ct = 1:n_cutoffs
    y_preds_col = y_preds(:, ct);
    ix_not_nan = ~isnan(y_preds_col);
    y_preds_col_not_nan = y_preds_col(ix_not_nan);
    y_not_nan = y(ix_not_nan);
    
    auc = f_SampleError(y_preds_col_not_nan,  y_not_nan, 'AUC');
    AUCs(1, ct ) = auc;
end

stds = std(y_preds - repmat(y, 1, n_cutoffs));


end

