function [Y_predicteds]=f_main_MFG(X_train, y_train, X_test, ...
    model_type, nRP, r, groups)
% 2
% Random select features, build models, simply average a lot of models, no
%  model-instance association, no re-train, 
%  no top neighbours, no top models.
% Similar to random forest.

% sim: the similarity matrix.
% nRP: times of random permutation.
% r: number of feature groups
% 
% train, test: the index of train and test set.

% number of instance clusters -- this parameter is dummy

% ----------------------- train classifiers ------------------------------
[groups, mss] = f_train_models_MFG(X_train,y_train,...
    model_type, r, nRP, groups);

% ------------ Filter and trim classifiers for each instance -------------
% ----------------------- RE-train classifiers ----------------------------

% ------------------------- make predictions ------------------------------
% [Y_predicteds] = f_predict_FGC(X_test, groups, mss, ...
%     sim_sub, model_type,tfs, NN);
[Y_predicteds] = f_predict_MFG(X_test, groups, mss, model_type);

            
end

