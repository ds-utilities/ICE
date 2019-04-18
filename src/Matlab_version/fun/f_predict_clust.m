function [Y_pred] = f_predict_clust(X, ms, clus, sim, model_type,...
    tf, n_neighb)
% Make predictions by the ensemble-multi-cluster classifiers
% X is the testing data 
% ms are the models
%
% sim: is the similarity array between the test instances and the training
%   instances. By using sim, we can select the good classifiers among all
%   the trained classifiers. The main function is to select the neighbours
%   of each test instance. 
%  Each row correspond to a training instance, each column corresponds to a
%   test instance.
%  Usually, there are 3000 rows represent 3000 positive instances when
%   onlyPosi=1.
%
% model_type: classification algorithm. 
%   Options: 'linear_regress'


if nargin < 7
    % number of neighbors to use in the test stage. 
    n_neighb = 10;
end
if nargin<6
    tf = true(size(sim,1), 1);
end

% make predictions
Y_pred = zeros(size(X,1), 1);

% convert the cell format cluster information to an array
arr = f_clustLoc_cell_to_arr(clus);

% X is the testing data 
for node = 1:size(X, 1)
	% x_node is an testing instance
    x_node = X(node, :);
    % Euclidean distance scores. 
    %s = sim(ix, test_loc(node));
    s = sim(:, node);
    
    % ----------- For each test x, find its top neighbours ----------------
    % sort, find the good neighbors, s_id: ranked neighbour ids
    [~, s_id] = sort(s, 'descend');
    top_neighbs = s_id(1: n_neighb);
    tf_top_neighbs = tf(top_neighbs);
    
    % ------------------------ Gather top models --------------------------
    % if strcmp(model_type, 'linear_regress')
    %top_models = [];
    top_models = cell(n_neighb, 1);    
    for i=1:n_neighb % for the top 10 neighbours
        % tn_id: top_neighb_id
        tn_id = top_neighbs(i);
        % c_id: top_neighb_cluster_id
        c_id = arr(tn_id);
        % the classifier (model) for the cluster
        model = ms{c_id};
        % top_models{i}: model for the top neighbour
        top_models{i} = model;
        
    end
    
    
    ensemble_predicts = zeros(n_neighb, 1);
    if strcmp(model_type, 'linear_regress')
        % Convert the cell format model list to n row of models
        tmp = top_models;
        top_models = [];
        for i=1:n_neighb
            model = tmp{i};
            % Convert the model to column
            model = model(:);
            top_models = [top_models; model'];            
        end
        % Prediction
        for i=1:n_neighb
            if tf_top_neighbs(i) == 1
                b = top_models(i, :); % b is a row vector
                ensemble_predicts(i) = [x_node, 1] * b';
            else
                % If the neighbour instance does not 'belongs' to the
                %  feature group, just predict nan.
                ensemble_predicts(i) = nan;
            end
        end
        Y_pred(node, 1) = nanmean(ensemble_predicts(:));
        %node,
        
    end   

    
end


end