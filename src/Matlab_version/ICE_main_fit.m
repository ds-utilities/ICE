
addpath('./fun/');

pa = '../../data/data_all/';
useParfor = 1;
doNorm = 1;
adv_whole = 0.3;
adv_self = 0.5;
topNN = 10; % top n neighbours in prediction
% 
k_fold = 10;
% *****************************************************
I = 100; % number of trees each RF
%
for i =  49:-1:1
    %names{i},
    load([pa, int2str(i), '/data.mat', ]);
    % ----------------------------------------------
    
    [each_clus_szs] = f_clus_size( floor( size(X, 1)*0.9 ) );
    n_circles = length(each_clus_szs);
    %n_clus = 25;
    n_clus = ceil(10 * log10(size(X, 1)));
    %n_clus_cumu = n_clus * 4;
    n_clus_cumu = n_clus * n_circles;
    % % cluster size set to 3/4, 1/2, 1/4, 1/10;
    % ----------------------------------------------
    
    % sim for testing instances to find their neighbors
    sim = corrcoef(X');
    sim = sim-diag(diag(sim));
    
    ress = {};
    % the first col is for using the whole data, the second to the last-1
    %  cols are for using each cluster, and the last col is for using tactic;
    y_pred_all_methods = zeros(length(y), n_clus_cumu+2);
    [train, test] = f_Kfold_cv(y, k_fold);
    
    for k=1:k_fold
        % train and test index
        X_te = X(test{k}, :);
        y_te = y(test{k});
        X_tr = X(train{k}, :);
        y_tr = y(train{k});
        
        % ----------------------------------------------------
        clus = f_fuzzy_rwr_clusters_v3(X_tr, n_clus , each_clus_szs);
        
        % ----------------------------------------------------
        tfs = f_clus_to_tfs(clus, size(X_tr, 1) );
        
        % **************** calculate the decision table ***************
        %dec_mat=f_AB_dec_tab_3(X_tr, y_tr, clus, useParfor,adv_whole,adv_self);
        %[dec_mat, pred_mat]=f_dec_tab_4_bg_svm(X_tr, y_tr, clus, useParfor);
        [dec_mat, pred_mat]=f_dec_tab_4_rf(X_tr, y_tr, clus, useParfor, I);
        
        % ************* predict test, using each cluster **************
        % using train to predict test, using all clusters
        %y_pred_all_methods_1Fold = f_tr_te_all_clus_bg_svm(X_tr, y_tr, ...
        y_pred_all_methods_1Fold = f_tr_te_all_clus_RF(X_tr, y_tr, ...
            X_te, y_te, clus, doNorm, I);
        y_pred_all_methods(test{k}, 1:end-1) = y_pred_all_methods_1Fold;
        
        % ************* for tactic prediction *************
        % final tactic result
        y_pred_tac = zeros(size(X_te, 1), 1 );
        neighbour_mat = sim(train{k}, test{k});
        
        % ---------- for each testing instance ----------
        for j = 1:length(test{k})
            %for j = 1
            % find the top 10 neighbors for each test
            neighbour_col = neighbour_mat(:, j);
            [~, ix] = sort(neighbour_col, 'descend');
            ix_top_neighbors = ix(1:topNN);
            
            % ---------- find the neighbors' picks ----------
            %clus_ids_to_use = zeros(topNN,1);
            clus_ids_to_use = [];
            for cur_nb = 1:topNN % for each neighbor
                clus_id = find(tfs(ix_top_neighbors(cur_nb, :), :) );
                %clus_id,
                cur_nb_ix = ix_top_neighbors(cur_nb);
                
                %dec_row = dec_mat(clus_id, :);
                dec_row = dec_mat(cur_nb_ix, :);
                dec_row(:, end    ) = dec_row(:, end    ) - adv_whole;
                dec_row(:, clus_id) = dec_row(:, clus_id) - adv_self ;
                
                %[~, ix] = sort(dec_row, 'ascend');
                %clus_id_to_use = ix(1);
                % chose the cluster with the min error
                % just in case dec_row has multi rows
                clus_id_to_use = [];
                for ix_tmp = 1:size(dec_row,1) % for each dec row
                    %tmp = find(dec_row(ix_tmp,:) == min(dec_row(ix_tmp,:) ) );
                    % get the clusters that better than whole
                    tmp = find( dec_row(ix_tmp,:) > dec_row(ix_tmp, end) ) ;
                    
                    %tmp = tmp(1);
                    clus_id_to_use = [clus_id_to_use, tmp];
                end
                clus_ids_to_use = [clus_ids_to_use, clus_id_to_use];
                %clus_id_to_use,
            end
            
            % ---------- vote for its NO. 1 cluster ----------
            % look up the cluster in the decision table to choose a proper
            %  training set from whole and each cluster.
            clus_ids_to_use(clus_ids_to_use == (n_clus_cumu+1) ) = 0;
            clus_ids_to_use = clus_ids_to_use + 1;
            
            % use a good amount of whole
            clus_ids_to_use = [clus_ids_to_use, ones(1, topNN) ]; % ones is for whole
            y_pred_tac(j) = nanmean(y_pred_all_methods_1Fold(j, clus_ids_to_use));
            
        end
        y_pred_all_methods(test{k}, end) = y_pred_tac;
        
        ress{i,k+1} = {dec_mat, clus, tfs};
        
    end
    ress{i,1} = y_pred_all_methods;
    %
    % ------------ compute AUC for the whole predictions ------------
    aucs = zeros(1,3); % whole; whole with -x option; tactic
    % for whole
    aucs(1, 1) = f_SampleError(y_pred_all_methods(:, 1), y, 'AUC');
    % weka -x whole
    %[~, auc_x] = f_weka_RF_arff_k_fo(X, y, k_fold);
    %aucs(1, 2) = auc_x;
    
    % for tactic
    tmp = y_pred_all_methods(:, end);
    tmp(isnan(tmp)) = nanmean(tmp);
    aucs(1, 3) = f_SampleError(tmp, y, 'AUC');
    
    ress{i, k_fold+2} = aucs;
    ress{i, k_fold+3} = {train, test};
    
    save(['../../data/rf_I_',int2str(I),'/', int2str(i), '_', ...
        int2str(k_fold), 'fo.mat'], 'ress');
   
end





%%


%%
