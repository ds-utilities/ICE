function [clus, net] = f_do_clustering(X, clust_alg, q, simType)
% clust_alg: clustering algorithm used.
% q: number of clusters expected
%   If choose to use 'HQcut' clustering algorithm, the variable q is
%    useless, but we have to set if to some value, 0 is fine, even set to a
%    none string '' is fine.

% Kmeas is much faster than HQCut.

if nargin < 4
    simType = 'euclidean';
end
if nargin < 3
    q = 0;
end
if nargin < 2
    clust_alg = 'Qcut';
end

net = 0;
if strcmp(clust_alg, 'kmeans')
    %size(X),
    %q,
    
    % % clus = kmeans(X, q);
    %try
    %clus = kmeans(X, q, 'Distance', 'cosine');
    % % use cosine distance to balance the clusters. 
    %catch exception
    %    try
    %    clus = kmeans(X, q, 'Distance', 'correlation');
    %    catch exception
    %        clus = kmeans(X, q);
    %    end
    %end
    if strcmp(simType, 'cosine')
        clus = kmeans(X, q, 'Distance', 'cosine', 'Replicates', 100);
    elseif strcmp(simType, 'euclidean')
        clus = kmeans(X, q, 'Replicates', 100 );
    elseif strcmp(simType, 'corr')
        clus = kmeans(X, q, 'Distance', 'correlation', 'Replicates', 100 );
    end
    
    
    %clus = f_clustLoc_arr_to_cell(clus);
    
elseif strcmp(clust_alg, 'Qcut')
    [clus, net, ~, ~, ~, ~] = getAutoGeometricCommunity(X);
    
    
elseif strcmp(clust_alg, 'HQcut')
    [clust, net, ~, ~, ~, ~] = getAutoGeometricCommunity(X);
    clus = HQcut(net, clust);
else
    fprintf('errer. Only support clustering algorithm kmeans and HQcut\n');
end


end

