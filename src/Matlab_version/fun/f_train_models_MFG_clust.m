function [groups, mss, cluss] = f_train_models_MFG_clust(X,Y,model_type, ...
    clust_alg,q,onlyPosi, r, nRP)
% MFG: multiple feature-grouping, do feature grouping multiple times to
%  allow a good chance of good features go into one group.
%
% X: training data.
% Y: training class labels.
%    Note: we assume even positive and negative instances when onlyPosi=1
% model_type: classification algorithm. 
%   Options: 'linear_regress'
%
% clust_alg: clustering algorithm used. it can be 'kmeans' and 'HQcut'
% q: number of clusters expected. Don't need it when clust_alg is 'HQcut'.
% r: number of feature groups. Don't need it when clust_alg is 'HQcut'.
%
% nRP: times of random permutation.
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
% 
% Workflow:
% 1. Feature gouping, return the divided 'Xs' and the group infor.
% 2. For each X slice, do f_train_models_clust(), returns the models and
%   the cluster information. store them in mss and cluss.

%  1. Feature gouping
groups = [];
X_sets = [];
for i =1:nRP
    [xs, g] = f_divide_attri_rand_perm(X, r);
    groups = [groups; g];
    X_sets = [X_sets, xs];
end
%groups,

% train models on each X slice
mss = cell(1, length(groups));
cluss = cell(1, length(groups));
for i=1:length(groups)
    if length(groups{i}) >=3 % just to make sure the code is correct, 
        % probably will always run this branch.
        [ms, clus] = f_train_models_clust(X_sets{i},Y,...
            model_type,clust_alg,q,onlyPosi);
        % ms is just one model, clus is just all the instances.
        mss{1, i} = ms;
        cluss{1, i} = clus;
    else
        mss{1, i} = '';
        cluss{1, i} = '';
    end
    %fprintf('  group %d\n', i);
end
% mss is a list of models
% cluss is a list of instances, actually, since there is no clustering on
%  the instances, cluss is just a dummy variable. 




end