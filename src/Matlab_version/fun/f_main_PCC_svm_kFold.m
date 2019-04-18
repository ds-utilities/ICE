function [y_preds] = f_main_PCC_svm_kFold(pa, name, k_fold, n_models)
% 
% Example:
% f_main_svm_kFold('/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/balloons/', 'balloons', 10)

% load([pa, name, '/', name, '.mat']);
load([pa, name, '.mat']);

pa_rwNets = [pa, 'rwNets/'];
% pa_arff = [pa, name, '/', 'arff_kf/'];
pa_arff = [pa, 'arff_kf/'];
[~, test] = f_StratifiedKFold(length(y), k_fold);
% n_models = [1,2,5,10,20,30,40,50,100,150,200,300,400,500,600,700,800,900];

y_preds = zeros(length(y), length(n_models));

   
for fold = 1:k_fold        
    train_arff = [pa_arff, 'fo_',int2str(fold),'_tr.arff'];
    test_arff = [pa_arff, 'fo_',int2str(fold),'_te.arff'];
    rw_netF_prefix = [pa_rwNets, 'net_fo_', int2str(fold), '_id_'];

    [pred] = f_main_PCC_weka_svm(train_arff, test_arff, ...
        rw_netF_prefix, n_models);

    ix_test = test{fold};
    y_preds(ix_test, :) = pred;
    fold,
end


% ----------------------- Save y_preds to mat file -----------------------
pa_yPred = [pa, '/', 'yPred/'];
if exist(pa_yPred, 'dir') ~= 7
    mkdir(pa_yPred)
end

fn_yPred = [pa_yPred, 'PCC_svm.mat'];
save(fn_yPred, 'y_preds', 'n_models');

end

