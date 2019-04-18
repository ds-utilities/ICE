function [y_preds] = f_main_PCC_kFold(pa, name, k_fold, n_models, alg, ...
    suffix, train, test)
% alg can be 'svm' or 'rf'

if nargin < 6
    suffix = '';
end


% Example:
% f_main_svm_kFold('/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/balloons/', 'balloons', 10, 'rf')

% load([pa, name, '/', name, '.mat']);
load([pa, name, '.mat']); % in order to know #instances.

pa_rwNets = [pa, 'rwNets/'];
% pa_arff = [pa, name, '/', 'arff_kf/'];
pa_arff = [pa, 'arff_kf/'];
%[~, test] = f_StratifiedKFold(length(y), k_fold);
% load train and test
load([pa, 'arff_kf/k_fold_divi.mat']);

% n_models = [1,2,5,10,20,30,40,50,100,150,200,300,400,500,600,700,800,900];

y_preds = zeros(length(y), length(n_models));

   
for fold = 1:k_fold
    train_arff = [pa_arff, 'fo_',int2str(fold),'_tr.arff'];
    test_arff = [pa_arff, 'fo_',int2str(fold),'_te.arff'];
    rw_netF_prefix = [pa_rwNets, 'net_fo_', int2str(fold), '_id_'];

    if strcmp(alg, 'svm') == 1
     pred=f_main_PCC_weka_svm(train_arff,test_arff,rw_netF_prefix,n_models);
    elseif strcmp(alg, 'rf') == 1
     pred=f_main_PCC_rf(train_arff,test_arff,rw_netF_prefix,n_models);        
    end
    
    ix_test = test{fold};
    y_preds(ix_test, :) = pred;
    fold,
end


% ----------------------- Save y_preds to mat file -----------------------
pa_yPred = [pa, '/', 'yPred/'];
if exist(pa_yPred, 'dir') ~= 7
    mkdir(pa_yPred)
end
fn_yPred = [pa_yPred, 'PCC_',alg, suffix, '.mat'];

% -----------------------------------
% in order to keep the previous result in the same file.
y_preds_1 = y_preds;
n_models_1 = n_models;
try
    load(fn_yPred);
catch exception
    y_preds = [];
    n_models = [];
end
y_preds = [y_preds_1, y_preds];
n_models = [n_models_1, n_models];
% -----------------------------------

save(fn_yPred, 'y_preds', 'n_models');

end

