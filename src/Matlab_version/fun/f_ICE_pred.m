function [y_pred_ice, n_partials, n_wholes, n_unique_ms] = ...
    f_ICE_pred(X_te, dec_mat, topNN, adv_whole, adv_self, alpha, beta, ...
    tfs, y_pred_all_methods_1Fold, n_clus, neighbour_mat)
% ICE prediction.
% input
%   X_te, decision table and the parameters,
% get the prediction value

n_inst_te   = size(X_te, 1) ;
y_pred_ice  = zeros(n_inst_te, 1 );

% if more more neighbours than the data has, topNN = size(data)
if  topNN >  n_inst_te
    topNN =  floor( 0.98 * n_inst_te );
end

% ---------- for each testing instance ----------
n_partials  = zeros(size(y_pred_ice));
n_wholes    = zeros(size(y_pred_ice));
n_unique_ms = zeros(size(y_pred_ice));
for j = 1:n_inst_te
    % find the top neighbors for each test
    neighbour_col    = neighbour_mat(:, j);
    [~, ix]          = sort(neighbour_col, 'descend');
    ix_top_neighbors = ix(1:topNN);
    
    % ---------- find the neighbors' picks ----------
    clus_ids_to_use = [];
    for cur_nb      = 1:topNN
        clus_id     = find(tfs(ix_top_neighbors(cur_nb, :), :) );
        cur_nb_ix   = ix_top_neighbors(cur_nb);
        
        dec_row             = dec_mat(cur_nb_ix, :);
        dec_row(:, end    ) = dec_row(:, end    ) - adv_whole;
        dec_row(:, clus_id) = dec_row(:, clus_id) - adv_self;
        
        clus_id_to_use = [];
        for ix_tmp = 1:size(dec_row,1) % for each dec row
            % get the clusters that better than whole
            tmp            = find( dec_row(ix_tmp,:)<dec_row(ix_tmp, end) );
            clus_id_to_use = [clus_id_to_use ,       tmp];
        end
        clus_ids_to_use    = [clus_ids_to_use,       clus_id_to_use];
    end
    
    % ---------- vote for its NO. 1 cluster ----------
    % look up the cluster in the decision table to choose a proper
    %  training set from whole and each cluster.
    clus_ids_to_use(clus_ids_to_use == (n_clus+1) ) = 0;
    clus_ids_to_use = clus_ids_to_use + 1;
    n_partial       = length(clus_ids_to_use);
    n_partials(j)   = n_partial;
    n_unique_ms(j)  = length(unique(clus_ids_to_use))-1;
    
    % use a proper amount of whole    
    n_whole         = round(alpha.*length(clus_ids_to_use) + beta.*topNN);
    n_wholes(j)     = n_whole;
    
    clus_ids_to_use = [clus_ids_to_use,  ones(1,  n_whole) ];
    
    %size(y_pred_all_methods_1Fold),
    %j,
    %clus_ids_to_use,
    
    y_pred_ice(j)   = nanmean(y_pred_all_methods_1Fold(j, clus_ids_to_use));
end




end

