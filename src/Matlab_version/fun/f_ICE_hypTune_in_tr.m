function [max_auc, topNN, adv_whole, adv_self, alpha, beta, best_set] = ...
    f_ICE_hypTune_in_tr(dec_mat, X_tr, y_tr, parameters, ...
    tfs, y_pred_all_methods_1Fold, n_clus, neighbour_mat)

X_te              = X_tr;

aucs              = zeros(1, size(parameters, 2));
% f = fopen(['../data/s22_real_kfo_v4_2_rf_I_100/', int2str(data_ix),'.txt'], 'w');
for i             = 1:size(parameters, 2)
    topNN         = parameters(1, i);
    adv_whole     = parameters(2, i);
    adv_self      = parameters(3, i);
    alpha         = parameters(4, i);
    beta          = parameters(5, i);

    % ------------------------ ice prediction ----------------------------
    [y_pred_ice, n_partials, n_wholes, n_unique_ms] = ...
        f_ICE_pred(X_te, dec_mat, topNN, adv_whole, adv_self, alpha, beta, ...
        tfs, y_pred_all_methods_1Fold, n_clus, neighbour_mat);
    
    yp            = y_pred_ice;
    yp(isnan(yp)) = nanmean(yp);
    aucs(1, i)    = f_SampleError(yp, y_tr, 'AUC');
    
    % fprintf(f, '%.5f,  ', aucs(1, i));
end

% ----------------------- find the best parameters -----------------------
[~, ix]           = max(aucs);
ix                = ix(1);

max_auc           = aucs(ix);
topNN             = parameters(1, ix);
adv_whole         = parameters(2, ix);
adv_self          = parameters(3, ix);
alpha             = parameters(4, ix);
beta              = parameters(5, ix);

best_set          = parameters(:, ix);

end

