function [res] = f_best_combine_auc_of_data_RF(data_name)
% read the comperhensive result of bagging-svm and pcc-svm and the hybrid result.

% 
pa = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/';
[~, ~, auc_RF_200, auc_pcc_rf_200]=...
    f_compute_AUC_1_data_only_RF(pa, data_name);



pa2 = [pa, data_name, '/'];
if exist([pa2,data_name,'.arff'],'file')==2 && exist([pa2,'yPred/','PCC_rf.mat'],'file')==2
    
    load([pa2,'yPred/','RF.mat']);y_preds_RF=y_preds;n_models_RF=n_models;
    load([pa2,'yPred/','PCC_rf.mat']); y_preds_pcc_rf =y_preds;n_models_pcc_rf=n_models;
    
    load([pa2, data_name, '.mat']); % load X and y

    fprintf('\n');
    [auc_hy_bst,nMS1,nMS2,  auc_RF_bst,nMS_RF_bst, ...
        auc_pcc_rf_bst, nMS_pcc_rf_bst]=...
        f_best_y_pred_comb(n_models_RF, y_preds_RF, ...
        n_models_pcc_rf, y_preds_pcc_rf, y, ...
        'RF  ', 'PCC rf ');
end


res = [auc_RF_200, auc_pcc_rf_200, auc_pcc_rf_200-auc_RF_200,...
    auc_RF_bst, nMS_RF_bst, auc_pcc_rf_bst, nMS_pcc_rf_bst,...
    auc_hy_bst, nMS1,nMS2, auc_pcc_rf_bst-auc_RF_bst, ...
    auc_hy_bst-auc_RF_bst];


end

