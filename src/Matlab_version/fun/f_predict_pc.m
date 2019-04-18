function [Y_predicteds] = f_predict_pc(X, classifiers, sim_sub, NN, ...
    model_type, y_train, isWekaJava, useMATLAB)
% Make predictions by the personalized classifiers.
% X is the testing set
% sim: is the similarity array between the test instances and the training
%  instances. By using sim, we can select the good classifiers among all
%  the trained classifiers. The main function is to select the neighbours
%  of each test instance. 
%  Each row correspond to a training instance, each column corresponds to a
%   test instance.

% make predictions
Y_predicteds = zeros(size(X,1), 1);
%X_test = X(test_loc, :);

% number of neighbors to use in the test stage. 
% n_neighb_to_test = 100;

for node = 1:size(X, 1)
%for node = 1
	% x_node is an instance
    x_node = X(node, :);
    
    % similarity column
    s = sim_sub(:, node);    
    [~, s_id] = sort(s, 'descend');
    top_neighbs = s_id(1: NN);
    
    top_models = [];
    % put all the relavent models into a single cell array.
    for i=1:NN
        top_models = [top_models; classifiers{1, top_neighbs(i)}];
    end
    
    if isempty(top_models)
        % if there are no models by the instance's neighbours, just give
        %  it the label of its nearest neighbour's.
        y_train(top_neighbs(1)),
        Y_predicteds(node, 1) = y_train(top_neighbs(1));
    else
        % if there are some models of its top neighbours, do the regular
        %  prediction.
        ensemble_predicts = zeros(length(top_models), 1);
        for i=1:size(top_models, 1)
            m = top_models{i, :};
            % ensemble_predicts(i) = [x_node, 1] * m;
            ensemble_predicts(i) = f_predict_simple(x_node, m, model_type,...
                isWekaJava, useMATLAB);

        end
        Y_predicteds(node, 1) = mean(ensemble_predicts(:));
    end
    %node,
end
    
end

