function [ Y_predicteds ] = f_main_dagging(X_train,y_train, ...
    X_test, model_type, nStacks )
% 

% model_type = 'linear_regress';

% ------------------------------------------------------------
% train classifiers
[ms] = f_train_models_dagging(X_train,y_train,...
    model_type, nStacks);

% ------------------------------------------------------------
% make predictions
[Y_predicteds] = f_predict_bagging(X_test, ms, model_type);


            
end

