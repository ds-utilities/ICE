function [tfs]=f_feGrp_filter_NB_2(X, Y, groups, ...
    c_p, c_n, cnt_y_p, cnt_y_n, topMs)
% This version won't use the sm - soft-margin, it use the top models with
%  lowest error for instance-model association.

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

% % Note that usually Re-train filtering is much less stringent compare to
% %  instance-model association.
    
% ============ Associate each instance with its top models. =============
% Compute the squre error 
errs = (Y_preds - repmat(Y, 1, size(Y_preds, 2))).^2;
a = errs;

tfs = false(size(X,1), length(groups));
for i=1:size(X, 1)
    row = a(i, :);
    % sort, delete models with large error
    [~, ix] = sort(row, 'ascend');
    cutoff = row(ix(topMs));
    tfs(i, :) = row <= cutoff;
    % This way is better than:
    %   tfs(i, ix(1:topMs)) = true;
    % Since if there are some models rank the same, this is more fair. Thus
    %  there might be case that, for example, we want to choose top 4 
    %  models, but it returns 6 models.
end



end








