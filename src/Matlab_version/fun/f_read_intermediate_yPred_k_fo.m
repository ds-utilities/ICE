function f_read_intermediate_yPred_k_fo(pa,k_fold,n_model,alg,suffix,...
    n_inst, pa_tmp)
% 
% alg = 'rf';
% suffix = 'mRank';

% pa_tmp = ['/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/2_ref/',...
%     'Jamiul_code/ZA/'];

% load([pa, name, '.mat']); % in order to know #instances.
% pa_rwNets = [pa, 'rwNets/'];
% pa_arff = [pa, 'arff_kf/'];
[~, test] = f_StratifiedKFold(n_inst, k_fold);


predss = zeros(n_inst, n_model);
nb_self_predss = zeros(n_inst, n_model);
nb_class_labels = zeros(n_inst, round(n_model/5) );

nb_idss = zeros(n_inst, round(n_model/5) );
nb_distancess = zeros(n_inst, round(n_model/5) );
nb_tfss = zeros(n_inst, n_model);


for fold = 1:k_fold
    ix_test = test{fold};
    
    fn_inter_res = [pa_tmp, 'preds_fo_', int2str(fold), '.mat'];
    load(fn_inter_res);
    
    predss(ix_test, :) = preds;
    nb_self_predss(ix_test, :) = nb_self_preds;
    nb_class_labels(ix_test, :) = nb_class_label;
    nb_idss(ix_test, :) = nb_ids;
    nb_distancess(ix_test, :) = nb_distances;
    nb_tfss(ix_test, :) = nb_tfs;
    
    system(['rm -f ', fn_inter_res]);
end

system(['rm -rf ', pa_tmp]);

% ----------------------- Save y_preds to mat file -----------------------
pa_yPred = [pa, '/', 'yPred/'];
if exist(pa_yPred, 'dir') ~= 7
    mkdir(pa_yPred)
end
fn_yPred = [pa_yPred, 'inter_PCC_',alg, suffix, '.mat'];

save(fn_yPred, 'predss', 'nb_self_predss', 'nb_class_labels', ...
    'nb_idss', 'nb_distancess', 'nb_tfss' );


% -----------------------------------





end

