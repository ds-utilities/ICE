function [tfs_retrain, rnks] = f_get_tfs_reTrain(Y_preds, Y, sm)
% 
% =========== For each model, find good instances for Re-train ============
% tfs_retrain: to decide which instances to re-train
tfs_retrain = true(size(Y_preds));
% rnks: Each column indicates for the model, the rank of the instances in
%  term of fitness. 
rnks = zeros(size(Y_preds));
for i=1:size(Y_preds, 2)
    [tf_col, rnk_col] = f_filter_insts_ranking(Y_preds(:, i), Y, sm);
    tfs_retrain(:, i) = tf_col;
    rnks(:, i) = rnk_col;
end



end

