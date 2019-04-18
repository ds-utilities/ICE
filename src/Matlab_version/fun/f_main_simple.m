function [ Y_predicteds ] = f_main_simple(X_train,y_train, ...
    X_test, model_type )
% 

% model_type = 'linear_regress';

% ------------------------------------------------------------
% train classifiers
m = f_train_model(X_train,y_train,model_type);

% ------------------------------------------------------------
% make predictions
[Y_predicteds] = f_predict_simple(X_test, m, model_type);

            
end

