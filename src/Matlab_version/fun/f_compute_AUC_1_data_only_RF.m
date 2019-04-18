function [aucs_RF, aucs_pcc_rf, auc_RF_200, auc_pcc_rf_200]=...
    f_compute_AUC_1_data_only_RF(pa, name)

    pa2 = [pa, name, '/'];
    %k_fold = 10;

    if exist([pa2, name, '.arff'], 'file') == 2       
        
        if exist([pa2, 'yPred/', 'PCC_rf.mat'], 'file') == 2
            load([pa2, name, '.mat']);
            
            load([pa2, 'yPred/', 'RF.mat']);       y_preds_RF = y_preds;
            aucs_RF = zeros(1, length(n_models));
            for i=1:size(n_models, 2)
                aucs_RF(i) = f_SampleError(y_preds_RF(:, i), y, 'AUC');
            end
            auc_RF_200 = aucs_RF(find(n_models == 200));
            
            load([pa2, 'yPred/', 'PCC_rf.mat']);  y_preds_pcc_rf = y_preds;
            aucs_pcc_rf = zeros(1, length(n_models));
            for i=1:size(n_models, 2)
                aucs_pcc_rf(i) = f_SampleError(y_preds_pcc_rf(:, i), y, 'AUC');
            end
            auc_pcc_rf_200 = aucs_pcc_rf(n_models == 200);
            
        end              
    end


end

