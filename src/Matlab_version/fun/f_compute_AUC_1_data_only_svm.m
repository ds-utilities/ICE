function [aucs_bagging_svm, aucs_pcc_svm, auc_bagging_svm_200, auc_pcc_svm_200]=...
    f_compute_AUC_1_data_only_svm(pa, name)

    pa2 = [pa, name, '/'];
    %k_fold = 10;

    if exist([pa2, name, '.arff'], 'file') == 2       
        
        if exist([pa2, 'yPred/', 'PCC_svm.mat'], 'file') == 2
            load([pa2, name, '.mat']);
            
            load([pa2, 'yPred/', 'BGsvm.mat']);  
            % Need to cut the array since all result is dumped here. 
            ix = find(n_models == 1); ix = ix(end);
            n_models = n_models(ix:end);
            y_preds = y_preds(:, ix:end);
            % -------------------------------------------------     
            y_preds_bagging_svm = y_preds;
            
            aucs_bagging_svm = zeros(1, length(n_models));
            for i=1:size(n_models, 2)
                aucs_bagging_svm(i) = f_SampleError(y_preds_bagging_svm(:, i), y, 'AUC');
            end
            auc_bagging_svm_200 = aucs_bagging_svm(find(n_models == 200));
            

            load([pa2, 'yPred/', 'PCC_svm.mat']);  
            y_preds_pcc_svm = y_preds;
            aucs_pcc_svm = zeros(1, length(n_models));
            for i=1:size(n_models, 2)
                aucs_pcc_svm(i) = f_SampleError(y_preds_pcc_svm(:, i), y, 'AUC');
            end
            auc_pcc_svm_200 = aucs_pcc_svm(n_models == 200);
            
        end              
    end


end

