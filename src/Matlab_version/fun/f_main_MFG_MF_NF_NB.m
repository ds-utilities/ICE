function [Y_predicteds]=f_main_MFG_MF_NF_NB(X_train, y_train, X_test, ...
    sim, NN, sm, groups, c_p, c_n, cnt_y_p, cnt_y_n )
% Associate the top models for each instance using soft-margin method.

% sim: the similarity matrix.
% NN: consider n top neighbours when do prediction
% 

% ----------------------- Train Classifiers ------------------------------
% The Naive Bayes model don't need to train models

% ------------ Associate each instance with top models -------------------
[tfs]=f_feGrp_filter_NB(X_train, y_train, groups,  ...
    c_p, c_n, cnt_y_p, cnt_y_n, sm);

% ----------------------- RE-train classifiers ----------------------------
% No re-train

% ---------------------------- Predictions --------------------------------
[Y_predicteds] = f_pred_MFG_nb_NB(X_test,  groups, ...
     c_p, c_n, cnt_y_p, cnt_y_n,  sim,  tfs, NN );

end

