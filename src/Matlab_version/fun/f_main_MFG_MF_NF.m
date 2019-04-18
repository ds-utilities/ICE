function [Y_predicteds]=f_main_MFG_MF_NF(X_train, y_train, X_test, ...
    model_type, sim, nRP, r, NN, sm, topMs, ...
    groups, methodCode)
% 
% sim: the similarity matrix.
% nRP: times of random permutation.
% r: number of feature groups
% NN: consider n top neighbours when do prediction
% 
% per: Filter and trim classifiers for each instance. percentage to get rid
%  of.
% topMs: using top topMs models for each instance. This parameter is
%  associated with per. If topMs is specified, then per should be 0. If per
%  is not set to 0, then topMs must be set to 0.
%
% onlyPosi: if 1, do clustering only on positive instances. Using option 1
%   when the negative instances are random generated, and assuming equal
%   size of positive instances and negative instances. 
%  If 0, do clustering on both positive and negative instnaces. Using this
%   option when both positive and negative instances are meaningful, and 
%   the negative instances are not random data.
% train, test: the index of train and test set.

% per_reTrain: the percentage of instances used when re-train the models
% ct_reTrain: the cutoff for the errs (errors) when doing reTrain, for each
%  model, instances with error greater than the cutoff will be deleted; the
%  rest instances will be used to reTrain.

% number of instance clusters -- this parameter is dummy
% q = 1;
% clust_alg: clustering algorithm -- this parameter is dummy
% clust_alg = 'kmeans';

% model_type = 'linear_regress';
%
% methodCode: could be 3, 4, 6, 7.
%  METHOD 3: do re-train=0, use tfs_retrain for model association.
%  METHOD 4: use sm->tfs_retrain for both association and re-train.
%  METHOD 6: use sm->tfs_retrain for Re-Train; use error rank for association
%  METHOD 7: use sm->tfs_retrain for Re-Train; use 'rank of rank' for association

% doReTrain: 1 - do re-train; 0 - no re-train, and use tfs_retrain for
%  model association -- METHOD 3.
if methodCode == 3
    doReTrain = 0;
else
    doReTrain = 1;
end

% ----------------------- Train Classifiers ------------------------------
[groups, mss] = f_train_models_MFG(X_train,y_train,...
    model_type, r, nRP,  groups);

% ------------ Associate each instance with top models -------------------
if methodCode == 7
    isRNK = 1; 
elseif methodCode == 6
    isRNK = 0;
else
    isRNK = 0;
end
[tfs, tfs_retrain]=f_feGrp_filter(X_train, y_train, model_type, ...
    groups, mss, sm, topMs, isRNK);

% ----------------------- RE-train classifiers ----------------------------
if doReTrain == 0
    % METHOD 3. NO RE-TRAIN
    % topMs is dummy para
    tfs = tfs_retrain;
else
    mss = f_train_models_MFG_reTrain(X_train, y_train, model_type, ...
         tfs_retrain, groups, mss);    
end

% ------------------------- make predictions ------------------------------
% % Y_predicteds = [];
if methodCode == 4
    tfs = tfs_retrain;
end

% if onlyPosi == 1
%     sim_sub = sim( train & (y==1), test);
% else
%     sim_sub = sim( train , test);
% end
[Y_predicteds] = f_predict_MFG_nb(X_test, groups, mss, ...
    sim, model_type, tfs, NN);


end

