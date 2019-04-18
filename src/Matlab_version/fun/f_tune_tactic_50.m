function [gains, perc_partial, avg_partial_counts, corrs, aucs_alls] = ...
    f_tune_tactic_50(data_all, names, fn_names, ...
    k_fold, n_clus,  topNN,  alpha, beta,  adv_whole,  adv_self)
% call by file s22_real_kfo_fuzzy_v2_new21_rf.m

gains = [];
n_partialsss = {};
n_wholesss = {};
n_unique_msss = {};
avg_unique_ms = zeros(49,1);
aucs_alls = {};

aucs_all = zeros(length(data_all), 4); % 3 base line AUC, and 1 ice AUC.
% aucs_all_35(:, 1:3) = base_aucs_allData;
for i = 1:length(data_all)  % for each data set
    %aucs_gain_mat = zeros(length(adv_wholes), length(adv_selfs));
    i,
    names{i},
    
    load(fn_names{i});
    % get rid of the empty cells
    i_flag = 1;
    for ix1 = 1:size(ress, 1)
        if length(ress{ix1,1} ) ~= 0
            i_flag = ix1;
        end
    end
    ress = ress(i_flag, :);
    
    
    X = data_all{i, 1}{1};
    y = data_all{i, 1}{2};
    % sim for testing instances to find their neighbors
    sim = corrcoef(X');
    sim = sim-diag(diag(sim)); % set the diag value to 0.
    
    train = ress{1, k_fold+3}{1};
    test = ress{1, k_fold+3}{2};
    y_pred_all_methods = ress{1, 1};
    
    n_partialss = zeros(size(y_pred_all_methods, 1), 1);
    n_wholess = zeros(size(y_pred_all_methods, 1), 1);
    n_unique_mss = zeros(size(y_pred_all_methods, 1), 1);
    %ress = {};
    % the first col is for using the whole data, the second to the last-1
    %  cols are for using each cluster, and the last col is for using ice;
    %y_pred_all_methods = zeros(length(y), n_clus+2);
    %[train, test] = f_Kfold_cv(y, k_fold);
    
    for k=1:k_fold
        %for k=1
        % train and test index
        X_te = X(test{k}, :);
        y_te = y(test{k});
        X_tr = X(train{k}, :);
        y_tr = y(train{k});
        
        dec_mat = ress{1,k+1}{1};
        clus = ress{1,k+1}{2};
        tfs = ress{1,k+1}{3};
        
        y_pred_all_methods_1Fold = y_pred_all_methods(test{k}, 1:end-1);
        neighbour_mat = sim(train{k}, test{k});
        
        [y_pred_ice, n_partials, n_wholes, n_unique_ms] = ...
            f_ICE_pred(X_te,dec_mat,topNN,adv_whole,adv_self,alpha,beta,...
            tfs, y_pred_all_methods_1Fold, n_clus, neighbour_mat);
        
        y_pred_all_methods(test{k}, end) = y_pred_ice;
        n_partialss(test{k}) = n_partials;
        n_wholess(test{k}) = n_wholes;
        n_unique_mss(test{k}) = n_unique_ms;
    end

    % ------------ compute AUC for the whole predictions ------------
    aucs = zeros(1,2); % whole; ice
    
    % for ice
    tmp = y_pred_all_methods(:, end);
    tmp(isnan(tmp)) = nanmean(tmp);
    auc_ice = f_SampleError(tmp, y, 'AUC'),
    
    
    aucs_all(i, 4) = auc_ice;
    % hybrid result
    aucs_all(i, 3) = f_SampleError(mean([tmp, y_pred_all_methods(:,1)], 2), y, 'AUC');
    
    % auc for whole
    aucs_all(i, 1) = ress{1, k_fold+2}(1,1);
    
    n_partialsss = n_partialss;
    n_wholesss = n_wholess;
    n_unique_msss = n_unique_mss;
    avg_unique_ms(i) = mean(n_unique_mss);
    mean(n_unique_mss),
    
end

gain = aucs_all(:,4)-aucs_all(:, 1);
gains = [gains gain];
aucs_alls = aucs_all;


perc_partial = zeros(size(n_partialsss));
for i=1:size(n_partialsss, 1)
    for j = 1:size(n_partialsss, 2)
        perc_partial(i, j) = mean(n_partialsss{i, j}) ./ mean(n_wholesss{i, j});
    end
end
perc_partial(isnan(perc_partial)) = 0;
%
avg_partial_counts = zeros(size(n_partialsss));
for i=1:size(n_partialsss, 1)
    for j = 1:size(n_partialsss, 2)
        avg_partial_counts(i, j) = mean(n_partialsss{i, j});
    end
end
avg_partial_counts(isnan(avg_partial_counts)) = 0;
avg_unique_ms,
display(['average # of models/ inst = ', num2str(mean(avg_unique_ms))]);
%


corrs = NaN;

end

