% ICE + fuzzy rwr with multiple neighborhoods
load(['../../data/new_49_names_and_ix.mat'])

%%
addpath('./fun/');
pa = '../../data/data_all/';
data_all50 = {};
for i = 1:49
    load([pa, int2str(i), '/data.mat']);
    data_all50{i, 1} = {X, y};
end

name_all50 = new_names(ix_inst);

k_fold = 10;
full_fns_50 = {};
pa = ['../../data/rf_I_100/'];
% pa = ['../data/s22_real_kfo_v4_rf_I_100_c400/'];
for i=1:length(name_all50)
    full_fns_50{i, 1} = [pa, int2str(i), '_10fo.mat'];
end


%%

n_Replicates = 100;
useParfor = 1;
doNorm = 0;
pa = ['../../data/'];
% ----------------------------------------------------
% n_clus = 25;
n_clus = 0;
% n_clus = 400;

topNNs = [];
alphas = 0.1:0.3:3;
betas = 0.1:0.3:3;
% adv_whole = 0.3;
adv_selfs = 0:0.05:0.8;  % dummy 
% adv_wholes = 0:0.1:0.8;
adv_whole = 1;   % dummy 

%%
alpha = 1;
beta  = 10; % amount whole f actor based on N (topNN)
adv_wholes = .3;
adv_self = .5;
topNN = 10;
%% 
[gains, perc_partial, avg_partial_counts, corrs, aucs_all] = ...
    f_tune_tactic_50(data_all50, name_all50, full_fns_50, ...
    n_Replicates, useParfor, ...
    doNorm, k_fold, n_clus, topNNs, topNN, alphas, alpha, ...
    betas, beta, adv_wholes, adv_whole, adv_selfs, adv_self);
%
base = aucs_all{1,1}(:, 1);
gain = gains(:, 1);
base = base(ix_inst);
gain = gain(ix_inst);

rand_comp(1:49, 8) = gain(1:49);
rand_comp(1:49, 9) = base(1:49);
%
ice = base + gain;
mean(ice),   ice = [ice ; mean(ice)];
%%
figure,  f_myScatter_4_tmp(base, gain, 'base AUC', 'AUC gain', 1:49);
xlim([0.4, 1.05]);

%%

figure,
f_myScatter_4_tmp(ice, ice_400-ice, 'ICE 100 AUC', 'ICE 400 AUC gain', 1:49);
xlim([0.4, 1.05]);


%%


%%
% Random Forest 10k trees.


I = 10000;
allData_aucs = [];
allData_pred = cell(1, 49);
for i = 1:49
    %n_clus = 100;
    load([pa, int2str(i), '/data.mat', ]);
    %sim = corrcoef(X');
    %sim = sim-diag(diag(sim));
    %ress = {};
    %y_pred_all_methods = zeros(length(y), n_clus+2);
    y_pred_all = zeros(length(y));
    [train, test] = f_Kfold_cv(y, k_fold);
    
    for k=1:k_fold
        % train and test index
        X_te = X(test{k}, :);
        y_te = y(test{k});
        X_tr = X(train{k}, :);
        y_tr = y(train{k});
        
        %[clus] = f_fuzzy_rwr_clusters(X_tr, n_clus);
        %tfs = f_clus_to_tfs(clus, size(X_tr, 1) );
        % y predictions using adaboost
        y_pred = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te, I); 
        y_pred_all(test{k}, 1) = y_pred;
        
    end
    allData_pred{1,i} = y_pred_all;
    i
    auc = f_SampleError(y_pred_all, y, 'AUC'),
    allData_aucs(i, 1) = auc;
end
%%
save(['../../data/s22_kfo_RF_I_',int2str(I),'_noICE.mat'], ...
    'allData_aucs', 'allData_pred', 'I', 'k_fold');



%%
load('../../data/s22_kfo_RF_I_10000_noICE.mat')
load(['../../data/new_49_names_and_ix.mat'])

%%
rf_10k = allData_aucs(ix_inst);

%%


%%


