function [ idxs, Xss, yss, modelss ] = f_soft_clusters1( X, y, n_clusts_max )

% Make 5 levels of clusters: 
%  level 0: 2^0 = 1 cluster, 
%  level 1: 2^1 = 2 clusters
%  level 2: 2^2 = 4 clusters
%  level 3: 2^3 = 8 cluster
%  level 4: 2^4 = 16 clusters
% For each level, get counting tables.
% Then for each instance Xi, it belongs to 5 clusters from each level, 
%  for each level, it has some best features.

% n_clusts_max = 4;

% idxs = {};
idxs = zeros(size(X, 1),  n_clusts_max+1 );

Xss = {};
yss = {};
modelss = {};
for m = 0:n_clusts_max
    % n_clusts = 2;
    n_clusts = 2^m;
    
    idx = kmeans(X, n_clusts);
    Xs = {};
    ys = {};
    
    % the counting tables.
    models = {};
    for i=1:n_clusts
        X_clust = X(idx==i, :);
        y_clust = y(idx==i);
        Xs{i} = X_clust;
        ys{i} = y_clust;
        
        [c_p, c_n, cnt_y_p, cnt_y_n ] = f_cnt_freq_NB(X_clust, y_clust);
        models{i, 1} = c_p;
        models{i, 2} = c_n;
        models{i, 3} = cnt_y_p;
        models{i, 4} = cnt_y_n;
    end
    
    % idxs{m+1} = idx;
    idxs(:, m+1) = idx;
    
    Xss{m+1} = Xs;
    yss{m+1} = ys;
    modelss{m+1} = models;
    
end



end

