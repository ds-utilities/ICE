function [tfs]=f_feGrp_filter_NB(X, Y, groups, ...
    c_p, c_n, cnt_y_p, cnt_y_n, sm)


% n: #models to be deleted
% n = length(groups) - topMs;

% =================== 1. predict y for each X_slice =======================
Y_preds = zeros(size(X,1), length(groups));
for i=1:length(groups)
    [Y_pred, ~]=f_predict_self_NB(c_p(:, groups{i}), c_n(:, groups{i}), ...
        cnt_y_p, cnt_y_n,  X(:, groups{i}));
    Y_preds(:, i) = Y_pred;

end

% =========== For each model, find good instances for Re-train ============
% tfs_retrain: to decide which instances to re-train
tfs = true(size(X,1), length(groups));
% rnks: Each column indicates for the model, the rank of the instances in
%  term of fitness. 

% siz = size(Y_preds)
% length(groups)
for i=1:length(groups)
    tf_col = f_filter_insts_fast(Y_preds(:, i), Y, sm);
    
    tfs(:, i) = tf_col;
end
%
% % Note that usually Re-train filtering is much less stringent compare to
% %  instance-model association.

% % tfs = tfs_retrain;



% ------ if one X_slice is empty, just delete all the column of tfs -------
% for i=1:length(groups)
%     if isempty(mss{i})
%         tfs(:, i) = false;
%     end
% end

end








