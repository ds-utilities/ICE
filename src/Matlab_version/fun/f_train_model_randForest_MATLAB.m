function [model] = f_train_model_randForest_MATLAB(X,y, nTrees)
% train model random forest MATLAB


model = TreeBagger(nTrees, X, y, ...
    'Method','classification',...
    'NumPredictorsToSample', round(log2(length(y)) ) +1 );


end

