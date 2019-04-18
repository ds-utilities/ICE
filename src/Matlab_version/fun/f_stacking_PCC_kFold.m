function [y_preds, ...
    predss, nb_self_predss, nb_class_labels,nb_idss,nb_distancess,nb_tfss]...
    = f_stacking_PCC_kFold(pa2, name, k_fold, n_model, alg, suffix, n_trees)
% 
% k_fold = 10;
% n_models = 500;
% alg = 'rf';
% suffix = 'mRank';
% n_trees = 100;

pa_tmp = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/',...
    'Jamiul_code/ZA/', int2str(randi(10000)), '/'];
mkdir(pa_tmp);

load([pa2, name, '.mat']);
pa_rwNets = [pa2, 'rwNets/'];
pa_arff = [pa2, 'arff_kf/'];
% [~, test] = f_StratifiedKFold(n_inst, k_fold);


for fold = 1:k_fold
    train_arff = [pa_arff, 'fo_',int2str(fold),'_tr.arff'];
    test_arff = [pa_arff, 'fo_',int2str(fold),'_te.arff'];
    rw_netF_prefix = [pa_rwNets, 'net_fo_', int2str(fold), '_id_'];

    fn_out = [pa_tmp, 'preds_fo_', int2str(fold), '.mat'];
%     if strcmp(alg, 'svm') == 1
%      pred=f_main_PCC_weka_svm(train_arff,test_arff,rw_netF_prefix,n_models);
%     elseif strcmp(alg, 'rf') == 1
     f_PCC_rf_stacking(train_arff,test_arff,rw_netF_prefix,...
         n_model,n_trees, fn_out);        
%     end
    
%     ix_test = test{fold};
%     y_preds(ix_test, :) = pred;
    fold,
end

% -----------------------------------------------------------
% read the 10 fold separated files into one prediction file in .mat format.
f_read_intermediate_yPred_k_fo(pa2,k_fold,n_model,alg,suffix,length(y),...
    pa_tmp);

pa_yPred = [pa2, '/', 'yPred/'];
fn_yPred = [pa_yPred, 'inter_PCC_',alg, suffix, '.mat'];
load(fn_yPred);

y_preds = nanmean(predss, 2);



end

