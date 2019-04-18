function [Y_pred] = f_predict_clust_self(X, y, ms, clus, model_type, onlyPosi)
% Revised from f_predict_clust(). This function won't predict the test
%  instances, rather, it just predict the training data, just for
%  self evaluation, in order to remove some bad classifiers in some
%  X_slice. Here the input X should be a X_slice.

Y_pred = zeros(size(X,1), 1);

% convert the cell format cluster information to an array
arr = f_clustLoc_cell_to_arr(clus);

if ~isempty(ms)
    if onlyPosi == 1
        % when negative instances are randomly generated
        X_posi = X(y==1, :);
        X_nega = X(y==0, :);

        % will store the predicted positive and negative y values
        Y_pred_posi = zeros(size(X,1)/2, 1);
        Y_pred_nega = zeros(size(X,1)/2, 1);

        for node = 1:size(X, 1)/2
            % x_node_posi is an instance
            x_node_posi = X_posi(node, :);
            x_node_nega = X_nega(node, :);
            % find the corresponding model
            m = ms{arr(node)};        
            Y_pred_posi(node,1)=f_predict_simple(x_node_posi,m,model_type);
            Y_pred_nega(node,1)=f_predict_simple(x_node_nega,m,model_type);        
        end
        Y_pred(y==1, 1) = Y_pred_posi;
        Y_pred(y==0, 1) = Y_pred_nega;    
    else
        % When negative instances are real world data.
        for node=1:size(X,1)
            % x_node is an instance
            x_node = X(node, :);
            m = ms{arr(node)};
            Y_pred(node,1)=f_predict_simple(x_node, m, model_type);
        end
    end
end


end



