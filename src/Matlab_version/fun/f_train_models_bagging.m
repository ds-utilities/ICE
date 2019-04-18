function [ms] = f_train_models_bagging(X,Y,model_type, n, per)
% X: training data
% Y: training class labels
% model_type: classification algorithm. 
%   Options: 'linear_regress'
%
% n: number of sampling times, e.g.: 10
% per: sampling percentage of the original size., e.g.: 0.9
%
% mss: models, size 1 by n

ms = cell(1, n);
% number of instances
ni = size(X, 1);
% number of instance been chosen each time
nch = round(ni*per);

for i=1:n
    locs = randperm(ni);
    % instances been chosen
    ch = locs(1:nch);
    X_sub = X(ch, :);
    Y_sub = Y(ch, :);
    
    % start to train a model
    ms{i} = f_train_model(X_sub, Y_sub, model_type);  
end

end

