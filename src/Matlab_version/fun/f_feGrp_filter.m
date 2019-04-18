function [tfs, tfs_retrain]=f_feGrp_filter(X, Y, model_type, ...
    groups, mss, sm, topMs, isRNK)
% Revised from f_feGrp_filter, when testing the models for each
%  instance, this version considers the neighbour instances. 
% 
% Feature group filtering, remove bad feature groups.
% 
% X: training data.
% Y: training class labels.
%    Note: we assume even positive and negative instances when onlyPosi=1
% model_type: classification algorithm. 
%   Options: 'linear_regress'
%
% onlyPosi: if 1, do clustering only on positive instances. Using option 1
%   when the negative instances are random generated, and assuming equal
%   size of positive instances and negative instances. 
%  If 0, do clustering on both positive and negative instnaces. Using this
%   option when both positive and negative instances are meaningful, and 
%   the negative instances are not random data.
%
% groups: cell array, n by 1, each element contains the feature ids 
%  (location) of the group. One feature belongs to only one group, no
%  overlaps.
% mss: stores the models, cell array, 1 by m groups, each element is a cell 
%  array, with size 1 by k clusters of the X_slice, each element is a
%  model.
% cluss: stores the cluster information of each X_slice, cell array, with
%  size 1 by m groups, each element is a cell array, with size 1 by k
%  clusters of the X_slice, each element corresponds to a cluster, contains
%  the instance ids of the cluster. 
% per: percentage of classifiers to be deleted.
% topMs: using top topMs models for each instance. This parameter is
%  associated with per. If topMs is specified, then per should be 0. If per
%  is not set to 0, then topMs must be set to 0.
%
% Every instance is associated with n (e.g. 10) classifiers (X-slice). For
%  each instance, we test each classifier and remove bad classifiers.
% 
% Returns tfs: a logical array with size n_instance by m_classifiers. Each 
%  element indicates if each classifier will be kept or not for the 
%  instance. 1(true) for keep, 0(false) for discard. 


% sm: soft margin for deciding which instance to re-train, see 
%  f_filter_insts() for reference

% topMs: #models to be associate with each instance.
% isRNK: 1 - use 'rank of rank' to select the top models of each instance. 
%        0 - use 'rank of err' of each instance, to select the top models. 
%         (But the problem is that the errors in each row of errs are not
%         consistent)
if nargin < 8
    isRNK = 1;
end


% n: #models to be deleted
% n = length(groups) - topMs;

% =================== 1. predict y for each X_slice =======================
Y_preds = zeros(size(X,1), length(groups));
for i=1:length(groups)
    %Y_preds(:, i) = f_predict_clust_self(X(:, groups{i}), Y, ...
    %    mss{i}, cluss{i}, model_type, onlyPosi);
    Y_preds(:, i) = f_predict_self(X(:, groups{i}), mss{i}, model_type);
end

% % -------------- if using linear-regress, normalize the data. -------------
% % Note that the normalization is done in the above prediction stage.


% =========== For each model, find good instances for Re-train ============
% tfs_retrain: to decide which instances to re-train
tfs_retrain = true(size(X,1), length(groups));
% rnks: Each column indicates for the model, the rank of the instances in
%  term of fitness. 
rnks = zeros(size(X,1), length(groups));
for i=1:length(groups)
    [tf_col, rnk_col] = f_filter_insts_ranking(Y_preds(:, i), Y, sm);
    tfs_retrain(:, i) = tf_col;
    rnks(:, i) = rnk_col;
end
%
% % Note that usually Re-train filtering is much less stringent compare to
% %  instance-model association.

% % tfs = tfs_retrain;

if isRNK == 1
    a = rnks;
else    
    % % Compute the squre error 
    errs = (Y_preds - repmat(Y, 1, size(Y_preds, 2))).^2;
    a = errs;
end
% ============ Associate each instance with its top models. =============
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


% ------ if one X_slice is empty, just delete all the column of tfs -------
for i=1:length(groups)
    if isempty(mss{i})
        tfs(:, i) = false;
    end
end

end








