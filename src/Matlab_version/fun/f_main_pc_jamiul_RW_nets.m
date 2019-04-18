function Y_predicteds = f_main_pc_jamiul_RW_nets(X_train, y_train, ...
    X_test, model_type, sim, NN, ix_train, ix_test, y, err_lim, isWekaJava, useMATLAB)
% 
% NN is the number of neighbours to use in prediction. Default NN = 100;

% err_lim: limitation for a prediction error in the training step. As
%  default, the error cannot be larger than 0.3.

% if nargin < 12
%     useMATLAB = 1;
% end
% if nargin < 11
%     % default using Weka command line and files.
%     isWekaJava = 0;
% end


% ------------------------ Train Classifiers -----------------------------
[classifiers, ~] = f_train_models_pc_jamiul_RW_nets(...
    X_train, y_train, ix_train, model_type, err_lim, isWekaJava, useMATLAB);

% ------------------------- Make Predictions -----------------------------
sim_sub = sim(ix_train, ix_test);

[Y_predicteds] = f_predict_pc(X_test, classifiers, sim_sub, ...
    NN, model_type, y_train, isWekaJava, useMATLAB);


% -------------------------- Delete Models -------------------------------
if useMATLAB == 0
    if ~strcmp(model_type, 'linear_regress') && isWekaJava==0
        f_delete_weka_models(classifiers, model_type, 'pc');
    end
end


end

