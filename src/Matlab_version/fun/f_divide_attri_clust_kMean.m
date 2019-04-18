function [X_sets, groups] = f_divide_attri_clust_kMean(X, q)
% Gives a data set X, with size m by n, divide X to several columns by
%  clustering the attributes. 
% q is the number of attribute groups 

if nargin < 2
    % number of groups
    q = 10;
end

% number of attributes
n = size(X, 2);

% size(X'),
% q,
groups = kmeans(X', q);
groups = f_clustLoc_arr_to_cell(groups);

% ------------------------------------------------------------------------
% If some clusters have less than 2 attributes, just combine the small
%  clusters into 1 cluster.
groups = f_shrink_clusters(groups);
% ------------------------------------------------------------------------

% divide X
X_sets = {};
for i=1:length(groups)
    X_sets{1, i} = X(:, groups{i});
end


end

