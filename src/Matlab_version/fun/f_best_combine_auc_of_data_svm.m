function [res] = f_best_combine_auc_of_data_svm(data_name)
% read the comperhensive result of bagging-svm and pcc-svm and the hybrid result.

% 
pa = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/';
[aucs_bagging_svm_all, aucs_pcc_svm_all, aucs_bagging_svm_200, aucs_pcc_svm_200]=...
    f_compute_AUC_1_data_only_svm(pa, data_name);



pa2 = [pa, data_name, '/'];
if exist([pa2,data_name,'.arff'],'file')==2 && exist([pa2,'yPred/','PCC_svm.mat'],'file')==2
    
    load([pa2,'yPred/','BGsvm.mat']);
    % Need to cut the array since all result is dumped here. 
    ix = find(n_models == 1); ix = ix(end);
    n_models = n_models(ix:end);
    y_preds = y_preds(:, ix:end);
    % -------------------------------------------------
    y_preds_bagging_svm=y_preds;n_models_bagging_svm=n_models;
    
    load([pa2,'yPred/','PCC_svm.mat']); y_preds_pcc_svm =y_preds;n_models_pcc_svm=n_models;
    
    %load([pa2, 'yPred/', 'PCC_rf.mat']); y_preds_pcc=y_preds; n_models_pcc=n_models;
    %load([pa2, 'yPred/', 'RF.mat']);   y_preds_rf = y_preds; n_models_rf=n_models;
    load([pa2, data_name, '.mat']); % load X and y

    fprintf('\n');
    [auc_hy_bst,nMS1,nMS2,  auc_bagging_svm_bst,nMS_bagging_svm_bst, ...
        auc_pcc_svm_bst, nMS_pcc_svm_bst]=...
        f_best_y_pred_comb(n_models_bagging_svm, y_preds_bagging_svm, ...
        n_models_pcc_svm, y_preds_pcc_svm, y, ...
        'bagging svm ', 'PCC svm ');
end


res = [aucs_bagging_svm_200, aucs_pcc_svm_200, aucs_pcc_svm_200-aucs_bagging_svm_200,...
    auc_bagging_svm_bst, nMS_bagging_svm_bst, auc_pcc_svm_bst, nMS_pcc_svm_bst,...
    auc_hy_bst, nMS1,nMS2, auc_pcc_svm_bst-auc_bagging_svm_bst, ...
    auc_hy_bst-auc_bagging_svm_bst];
end

