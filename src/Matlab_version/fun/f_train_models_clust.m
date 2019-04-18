function [ms, clus] = f_train_models_clust(X, Y, model_type, ...
    clust_alg, q, onlyPosi)
% Input a training data X and a list of class labels, including both
%  positive and negative instances (Y can be 0 or 1), train a list of
%  models.
%
% Workflow:
%  1. Clustering.
%  2. For each cluster, train a model
%  3. return the clustering information and the trained models for each
%    cluster.
%
% X: train data,
% Y: class labels
%    Note: we assume even positive and negative instances when onlyPosi=1
%
% model_type: classification algorithm. 
%   Options: 'linear_regress'
% 
% clust_alg: clustering algorithm used. it can be 'kmeans' and 'HQcut'
% q: number of clusters expected
%   note: since realizing choosing the type of clustering algorithm is just
%    time-consuming, I decide only using the HQcut clustering algorithm.
%    Using algorithms such as K-Mean need an extra parameter q, which add
%    more labor. I didn't remove the model_type and onlyPosi input since
%    there might be a high chance to change them later, but clust_alg using
%    'HQcut' is very common and I cannot think a reason using others
%    instead.
%   2016-3-3: now I have found a undeniable reason to keep the clust_alg
%    and q parameter and use Kmean instead of HQcut -- HQcut is too slow 
%    and KMean is just so fast. 
%   If choose to use 'HQcut' clustering algorithm, the variable q is
%    useless, but we have to set if to some value, 0 is fine, even set to a
%    none string '' is fine.

% onlyPosi: if 1, do clustering only on positive instances. Using option 1
%   when the negative instances are random generated, and assuming equal
%   size of positive instances and negative instances. 
%  If 0, do clustering on both positive and negative instnaces. Using this
%   option when both positive and negative instances are meaningful, and 
%   the negative instances are not random data.
%
% ms: trained models, cell array, size 1 by n, each element is a model
%  corresponding to the cluster.
% clus: cluster information, cell array, size 1 by n, each element stores
%  the instances ids of the cluster. 

ms = {};
if onlyPosi == 1
    % if the cluster is only for the positive instances
    X_posi = X(Y==1, :);
    
    % 1. Clustering
    clus = f_do_clustering(X_posi, clust_alg, q);
    
    % 2. For each cluster, train a model
    X_posi = X(Y==1, :);
    X_nega = X(Y==0, :);
    
    Y_posi = Y(Y==1, :);
    Y_nega = Y(Y==0, :);   
    
    for i=1:length(clus)
        % go over each cluster
        % positive instances for the current cluster
        X_posi_cur = X_posi(clus{i}, :);
        % negative instances for the current cluster
        X_nega_cur = X_nega(clus{i}, :);
        
        Y_posi_cur = Y_posi(clus{i}, :);
        Y_nega_cur = Y_nega(clus{i}, :);
        
        ms{i} = f_train_model([X_posi_cur; X_nega_cur],...
            [Y_posi_cur; Y_nega_cur],model_type);
        %fprintf('    cluster %d\n', i);
    end    
else
    % if clustering is on all the instances    
    % 1. Clustering
    clus = f_do_clustering(X, clust_alg, q);
    
    % 2. For each cluster, train a model    
    for i=1:length(clus)
        X_cur = X(clus{i}, :);
        Y_cur = Y(clus{i}, :);
        ms{i} = f_train_model(X_cur, Y_cur,model_type);
    end    
end

end

