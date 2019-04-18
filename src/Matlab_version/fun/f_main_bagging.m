function [ Y_predicteds ] = f_main_bagging(X_train,y_train, ...
    X_test, model_type, nSamp, per )
% 

% train classifiers
[ms] = f_train_models_bagging(X_train,y_train,...
    model_type, nSamp, per);

% ------------------------------------------------------------
% make predictions
[Y_predicteds] = f_predict_bagging(X_test, ms, model_type);
            
            
end

