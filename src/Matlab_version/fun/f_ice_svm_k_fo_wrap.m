function [ y_pred, auc_ice, auc_whole ] = f_ice_svm_k_fo_wrap(X, y, n_clus, k_fold, op)
% 

%adv_whole = 0.4;
%adv_self = 0.5;
adv_whole = 0.1;
adv_self = 0.9;

useParfor = 1;
doNorm = 1;
topNN = 10;

alpha1 = 0;
beta = 0;

% parameters of random walk with restart
each_clus_sz = round( size(X,1)/3 );
nSteps = 100;
lazi = 0.3;
do_rwr = 0;

% choose a good distance messurement

%   correlation
%sim = corrcoef(X');
%sim = sim-diag(diag(sim));

%   Euclidean distance
sim = squareform(pdist(X)); sim = -sim;

ress = {};
% the first col is for using the whole data, the second to the last-1
%  cols are for using each cluster, and the last col is for using tactic;
y_pred_all_methods = zeros(length(y), n_clus+2);
[train, test] = f_Kfold_cv(y, k_fold);

for k=1:k_fold
% for k=1
    %for k=1
    % train and test index
    X_te = X(test{k}, :);
    y_te = y(test{k});
    X_tr = X(train{k}, :);
    y_tr = y(train{k});

    % pick up a clustering method.
    %each_clus_sz = round( size(X,1)/3 );
    %nSteps = 200;
    %lazi = 0.5;
    %[clus] = f_fuzzy_rwr_clusters(X_tr, n_clus);
    clus=f_fuzzy_rwr_clusters(X_tr,n_clus,each_clus_sz,nSteps,lazi,do_rwr);
    
    % k-means
    %clus = f_do_clustering(X_tr, 'kmeans', n_clus),


    tfs = f_clus_to_tfs(clus, size(X_tr, 1) );

    % **************** calculate the decision table ***************
    %dec_mat=f_AB_dec_tab_3(X_tr, y_tr, clus, useParfor,adv_whole,adv_self);
    %[dec_mat, pred_mat]=f_dec_tab_4_bg_svm(X_tr, y_tr, clus, useParfor);
    [dec_mat, pred_mat]=f_dec_tab_4_svm(X_tr, y_tr, clus, useParfor,op);
    %dec_mat,
    
    % ************* predict test, using each cluster **************
    % using train to predict test, using all clusters
    %y_pred_all_methods_1Fold = f_tr_te_all_clus_bg_svm(X_tr, y_tr, ...
    y_pred_all_methods_1Fold = f_tr_te_all_clus_svm(X_tr, y_tr, ...
        X_te, y_te, clus, doNorm, op);
    y_pred_all_methods(test{k}, 1:end-1) = y_pred_all_methods_1Fold;

    % ************* for tactic prediction *************
    % final tactic result
    y_pred_tac = zeros(size(X_te, 1), 1 );
    neighbour_mat = sim(train{k}, test{k});
    for j = 1:length(test{k})
        %j,
        % find the top 10 neighbors for each test
        neighbour_col = neighbour_mat(:, j);
        [~, ix] = sort(neighbour_col, 'descend');
        ix_top_neighbors = ix(1:topNN);

        % ---------- find the neighbors' picks ----------
        %clus_ids_to_use = zeros(topNN,1);
        clus_ids_to_use = [];
        for cur_nb = 1:topNN
            clus_id = find(tfs(ix_top_neighbors(cur_nb, :), :) );
            %clus_id,
            %cur_nb,
            % fprintf('test\n');
            %ix_top_neighbors,% cur_nb,
            %cur_nb_ix = ix_top_neighbors(cur_nb, :),
            cur_nb_ix = ix_top_neighbors(cur_nb);

            dec_row = dec_mat(cur_nb_ix, :);
            dec_row(:, end    ) = dec_row(:, end    ) - adv_whole;
            dec_row(:, clus_id) = dec_row(:, clus_id) - adv_self;

            clus_id_to_use = [];
            for ix_tmp = 1:size(dec_row,1) % for each dec row
                %tmp = find(dec_row(ix_tmp,:) == min(dec_row(ix_tmp,:) ) );
                % get the clusters that better than whole
                tmp = find( dec_row(ix_tmp,:) < dec_row(ix_tmp, end) ) ;

                %tmp = tmp(1);
                clus_id_to_use = [clus_id_to_use, tmp];
            end
            clus_ids_to_use = [clus_ids_to_use, clus_id_to_use];
        end

        % ---------- vote for its NO. 1 cluster ----------
        % look up the cluster in the decision table to choose a proper
        %  training set from whole and each cluster.
        clus_ids_to_use(clus_ids_to_use == (n_clus+1) ) = 0;
        clus_ids_to_use = clus_ids_to_use + 1;
        n_partial = length(clus_ids_to_use);
        n_partials(j) = n_partial;
        n_unique_ms(j) = length(unique(clus_ids_to_use))-1;

        % use a good amount of whole
        %clus_ids_to_use = [clus_ids_to_use, ones(1, topNN) ]; % ones is for whole
        %clus_ids_to_use = [clus_ids_to_use, ones(1, topNN*10) ]; % ones is for whole

        n_whole = round(alpha1.*length(clus_ids_to_use) + beta.*topNN);
        n_wholes(j) = n_whole;

        clus_ids_to_use = [clus_ids_to_use, ...
            ones(1,  n_whole) ]; 
        %clus_ids_to_use = [clus_ids_to_use, ones(1, 1 + round(amount_whole_factor *length(clus_ids_to_use)) ) ]; % ones is for whole

        y_pred_tac(j) = nanmean(y_pred_all_methods_1Fold(j, clus_ids_to_use));
        unique(clus_ids_to_use),

        %alpha1 = amount_whole_factor;
    end
    y_pred_all_methods(test{k}, end) = y_pred_tac; % the last col is for ICE
    f_SampleError(y_pred_tac, y_te, 'AUC')
    f_SampleError(y_pred_all_methods_1Fold(:,1), y_te, 'AUC')
    k,
end

% for tactic
tmp = y_pred_all_methods(:, end);
tmp(isnan(tmp)) = nanmean(tmp);
auc_ice = f_SampleError(tmp, y, 'AUC')
y_pred = tmp;

% for whole
auc_whole = f_SampleError(y_pred_all_methods(:, 1), y, 'AUC')






end

