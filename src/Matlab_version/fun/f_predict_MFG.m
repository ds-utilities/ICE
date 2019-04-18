function [Y_predicteds] = f_predict_MFG(X, groups, mss, model_type)
% Compare to f_predict_FGC(), there is no model-instance association, no
%  top model selection, just average all the models to get a
%  randomForest-like result.

% Divide X
X_sets = f_groups2X_sets(X, groups);

% --------------------- Predict for each X slice -------------------------
% y_tmp: stores all the y predictions.
y_tmp = zeros(size(X,1), length(groups));
for i=1:length(groups)
    %Y_one_slice = nan(size(X, 1), 1);
    xs = X_sets{1, i};
    if ~isempty(mss{i})
        Y_one_slice = f_predict_simple(xs, mss{i}{1}, model_type);
        if ~strcmp(model_type, 'linear_regress')
            system(['rm -f ', mss{i}{1}]);
        end
    else
        fprintf('model is empty \n');        
    end
    y_tmp(:, i) = Y_one_slice;
    
end

Y_predicteds = nanmean(y_tmp, 2);



end

