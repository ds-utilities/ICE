function [y_preds] = f_main_RF_kFold(pa, name, k_fold, n_models)
% 
% Example:
% f_main_RF_kFold('/Volumes/Macintosh_HD/Users/zhengao/bio/3_ensembF/1_data/6_121_data/data/balloons/', 'balloons', 10)


% load([pa, name, '/', name, '.mat']);
load([pa, name, '.mat']);

% pa_arff = [pa, name, '/', 'arff_kf/'];
pa_arff = [pa, 'arff_kf/'];
[~, test] = f_StratifiedKFold(length(y), k_fold);
% n_models = [1,2,5,10,20,30,40,50,100,150,200,300,400,500,600,700,800,900];

y_preds = zeros(length(y), length(n_models));

for i = 1:length(n_models)
    n_model = n_models(i);
    for fold = 1:k_fold        
        train_arff = [pa_arff, 'fo_',int2str(fold),'_tr.arff'];
        test_arff = [pa_arff, 'fo_',int2str(fold),'_te.arff'];
        
        [pred] = f_main_RF_weka_file(train_arff, test_arff, n_model);
        ix_test = test{fold};
        y_preds(ix_test, i) = pred;
        fold,
    end
    i,
end


% ----------------------- Save y_preds to mat file -----------------------
pa_yPred = [pa, '/', 'yPred/'];
if exist(pa_yPred, 'dir') ~= 7
    mkdir(pa_yPred)
end
fn_yPred = [pa_yPred, 'RF.mat'];

% -----------------------------------
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

