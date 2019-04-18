function [ Y_predicteds ] = f_main_pc(X_train, y_train, ...
    X_test, model_type, sim, NN, nSteps, lazi,...
    high, low, ix_train, ix_test, y, err_lim)
% 
% nSteps: random walk steps
% nSteps = 1000;

% NN is the number of neighbours to use in prediction. Default NN = 100;


% lazi: random walk restart probability.
% lazi = 0.3;

% Compute the number of neighbors and the corresponding cutoffs
% high / low: average neighbor size's bound:
% high=7;
% low =2;

% model_type = 'linear_regress';
% err_lim: limitation for a prediction error in the training step. As
%  default, the error cannot be larger than 0.3.


% ------------------------------------------------------------
% train classifiers
[classifiers, ~] = f_train_models_pc(X_train,y_train,...
    sim, nSteps, lazi, high, low, ix_train, ix_test, y, model_type, err_lim);

% ------------------------------------------------------------
% make predictions
sim_sub = sim(ix_train, ix_test);

[Y_predicteds] = f_predict_pc(X_test, classifiers, sim_sub, ...
    NN, model_type, y_train);



            
end

