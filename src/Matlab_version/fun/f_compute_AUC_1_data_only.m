function [aucs_rf_1,aucs_pcc_1,aucs_hy_1]=f_compute_AUC_1_data_only(pa, name)

    pa2 = [pa, name, '/'];
    %k_fold = 10;

    if exist([pa2, name, '.arff'], 'file') == 2       
        
        if exist([pa2, 'yPred/', 'PCC_rf.mat'], 'file') == 2
            load([pa2, 'yPred/', 'PCC_rf.mat']);  y_preds_pcc = y_preds;
            load([pa2, 'yPred/', 'RF.mat']);       y_preds_rf = y_preds;

            load([pa2, name, '.mat']);
            fprintf('\nOn %s\n', name);
            for i=1:size(n_models, 2)
                a1 = f_SampleError(y_preds_rf(:, i), y, 'AUC');
                a2 = f_SampleError(y_preds_pcc(:, i), y, 'AUC');

                a3 = f_SampleError(mean([y_preds_rf(:, i) y_preds_pcc(:, i)], 2 ) , y, 'AUC');

                fprintf('n_models: %d   AUC RF: %.3f    PCC: %.3f    Combine: %.3f\n', n_models(i), a1, a2, a3);

                aucs_rf_1 = a1;
                aucs_pcc_1 = a2;
                aucs_hy_1 = a3;
            end
        
        
        end
    end


end

