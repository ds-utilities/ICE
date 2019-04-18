function [Y_pred] = f_predict_self(X, ms, model_type)
% Revised from f_predict_clust_self() and f_predict(). This function won't 
%  predict the test instances, rather, it just predict the training data, 
%  just for self evaluation, in order to remove some bad classifiers in 
%  some X_slice. Here the input X should be a X_slice.

Y_pred = zeros(size(X,1), 1);

% convert the cell format cluster information to an array
% arr = f_clustLoc_cell_to_arr(clus);
% e.g.: [1 1 1 1 2 3 3 3 3] 1 means cluster 1, 2 means cluster 2.

if ~isempty(ms)   
    Y_pred = f_predict_simple(X, ms{1}, model_type);
end


end



