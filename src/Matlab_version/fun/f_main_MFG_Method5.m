function [Y_predicteds]=f_main_MFG_Method5(X_train, y_train, X_test, ...
    model_type, sim, nRP, r, NN, sm1, sm2, groups)
% 
% sim: the similarity matrix.
% nRP: times of random permutation.
% r: number of feature groups
% NN: consider n top neighbours when do prediction
% 
% doReTrain: 1 - do re-train; 0 - no re-train, and use tfs_retrain for
%  model association -- METHOD 3.

% ----------------------- Train Classifiers ------------------------------
[groups, mss] = f_train_models_MFG(X_train,y_train,...
    model_type, r, nRP,  groups);

% ------------ Associate each instance with top models -------------------
[tfs, tfs_retrain]=f_feGrp_Method5(X_train, y_train, model_type, ...
    groups, mss, sm1, sm2);

% ----------------------- RE-train classifiers ----------------------------
mss = f_train_models_MFG_reTrain(X_train, y_train, model_type, ...
     tfs_retrain, groups, mss);    

% ------------------------- make predictions ------------------------------
[Y_predicteds] = f_predict_MFG_nb(X_test, groups, mss, ...
    sim, model_type, tfs, NN);

            
end

