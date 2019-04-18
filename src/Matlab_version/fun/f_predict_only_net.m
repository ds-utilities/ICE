function [Y_predicteds] = f_predict_only_net(X, net_sub, NN, ...
    y_train)
% Revised from f_predict_pc(), this time only use the similarity matrix /
%  the random-walk matrix to predict the labels. 
% 
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
    % similarity column
    column = net_sub(:, node);
    [~, ixs] = sort(column, 'descend');
    top_neighbs = ixs(1: NN);   
    
    %size(net_sub),
    %length(top_neighbs),
    %size(y_train),
    
    %  just give  its nearest neighbour's label.
    % y_train(top_neighbs(1));
    
    % or voting for the label.
    label = round(mean(y_train(top_neighbs) ));
    
    Y_predicteds(node, 1) = label;
    
end
    
end

