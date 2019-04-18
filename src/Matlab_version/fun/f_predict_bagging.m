function [Y_pred] = f_predict_bagging(X, ms, model_type)
% Prediction by bagging or by dagging.
% Note: this one is also used as dagging prediction.
% 
% X is the testing data 
% ms are the models
%
% model_type: classification algorithm. 
%   Options: 'linear_regress'


Y_pred = zeros(size(X,1), 1);

% 
Y_preds = zeros(size(X,1), length(ms));
if strcmp(model_type, 'linear_regress')
    % Convert the cell format model list to n row of models
    for i = 1:length(ms)
        m = ms{i};
        Y_preds(:, i) = f_predict_simple(X, m, model_type);
    end
    
end
Y_pred = nanmean(Y_preds, 2);

end

