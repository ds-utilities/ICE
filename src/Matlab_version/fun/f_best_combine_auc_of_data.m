function [aucBST, aucs_rf_200,aucs_pcc_200, auc_rf_bst, auc_pcc_bst] = ...
    f_best_combine_auc_of_data(data_name)
% read the comperhensive result of RF and pcc-rf and the hybrid result.

% 
pa = '/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/';
[aucs_rf_1,aucs_pcc_1,aucs_hy_1]=f_compute_AUC_1_data_only(pa, data_name);



pa2 = [pa, data_name, '/'];
if exist([pa2,data_name,'.arff'],'file')==2 && exist([pa2,'yPred/','PCC_rf.mat'],'file')==2
    
    load([pa2, 'yPred/', 'PCC_rf.mat']); y_preds_pcc=y_preds; n_models_pcc=n_models;
    load([pa2, 'yPred/', 'RF.mat']);   y_preds_rf = y_preds; n_models_rf=n_models;
    load([pa2, data_name, '.mat']); % load X and y

    fprintf('\n');
    [aucBST,nMS1,nMS2,  aucBST1only,nMS1only,  aucBST2only,nMS2only]=...
        f_best_y_pred_comb(n_models_rf, y_preds_rf, n_models_pcc, y_preds_pcc, y, ...
        'RF    ', 'PCC-RF');
end


aucs_rf_200 = aucs_rf_1(end);
aucs_pcc_200 = aucs_pcc_1(end);

auc_rf_bst = aucBST1only; % max(aucs_rf_1);
auc_pcc_bst = aucBST2only; % max(aucs_pcc_1);
end

