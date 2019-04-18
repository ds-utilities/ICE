function [ Y_predicteds ] = f_main_randForestMATLAB(X_train,y_train, ...
    X_test, nTrees)
% 

% train classifiers
m = f_train_model_randForest_MATLAB(X_train, y_train, nTrees);

% ------------------------------------------------------------
% make predictions
[Y_predicteds] = f_predict_randForest_MATLAB(X_test, m);



end

