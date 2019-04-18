function [ms] = f_train_models_dagging(X,Y,model_type, n)
% X: training data
% Y: training class labels
% model_type: classification algorithm. 
%   Options: 'linear_regress'
%
% n: number of sampling times, e.g.: 10
%
% mss: models, size 1 by n

ms = cell(1, n);
% number of instances
ni = size(X, 1);



p = randperm(ni);
% number of instances each group (stack)
a = ceil(ni/n);
% w is the real number of groups, while n is the rough estimate number of
%  groups.
w = ceil(ni/a);
groups = cell(w, 1);
for i=1:w-1
    groups{i,1} = p(a*(i-1) + 1: a*(i-1)+a);
end
groups{w,1} = p(a*(w-1) + 1: end);


for i=1:w
    %locs = randperm(ni);
    locs = groups{i};
    X_sub = X(locs, :);
    Y_sub = Y(locs, :);
    
    % start to train a model
    ms{i} = f_train_model(X_sub, Y_sub, model_type);  
end

end

