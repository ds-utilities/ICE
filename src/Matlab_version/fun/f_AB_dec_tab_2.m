function [dec_mat, y_pred_whole, auc_whole, y_pred_self, auc_self, ...
    y_pred_other, auc_other, y_pred_tactic, auc_tactic,...
    tactic_pick, tactic_pick_code] = ...
    f_AB_dec_tab_2(X, y, clus)
% This updated version also outputs the predicted values of y.
% support more than 3 clusters
% normalization take place in y_pred_self and y_pred_other, thus do not
%  need normalization when predict y_pred_tactic.
% ixsp is another cluster form.

adv_whole = 0.05;
adv_self = 0.02;

n_clusters = length(clus);
dec_mat = nan(n_clusters + 1, n_clusters);

%% --------------------------- WHOLE -------------------------------------
% Predict each cluster using the whole data. 
try
[y_pred_whole, auc_whole] = f_weka_RF_arff_10_fo_3(X, y);

for j=1:n_clusters        
    auc = f_SampleError(y_pred_whole(clus{j}), y(clus{j}), 'AUC');
    dec_mat(n_clusters+1, j) = auc;
end

fprintf('done using whole data to predict each clusters\n');
catch exception   
end
    

%% ----------------- SELF ---------------------------------------
% Self prediction using 10-fold CV
y_pred_self = nan(length(y), 1);

try
for j=1:n_clusters
    Xj = X(clus{j}, :) ;
    yj = y(clus{j}) ;
    dec_mat(j, j) = nan;

    if length(yj) > 15
        [y_pred, auc] = f_weka_RF_arff_10_fo_3(Xj, yj);
        dec_mat(j, j) = auc;
        
        % Normalization
        templete = y_pred_whole(clus{j});
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);
        
        % copy the normed prediction to the whole data.
        y_pred_self(clus{j}) = y_pred;

    end
end
fprintf('done self evaluation\n');
catch exception
end
    
y_tmp = y(~isnan(y_pred_self)); y_pred_self = y_pred_self(~isnan(y_pred_self));
auc_self = f_SampleError(y_pred_self, y_tmp, 'AUC');


%% ---------------------- Find best 'other' ------------------------------
% fill dec_mat for 'others'
% Using larger clusters to predict smaller clusters (e.g.: A predicts B)

try
for r = 1:n_clusters
    for c = r + 1 : n_clusters
        X_tr = X(clus{r}, :);
        y_tr = y(clus{r});            
        X_te = X(clus{c}, :);
        y_te = y(clus{c});
        dec_mat(r, c) = nan;

        % from training instances, random pick up instances equals to 
        %  90% test instances size for a fair comparison to 10-fold
        %  CV on testing data.
        %tr_rand_ix = randperm(length(y_tr));
        %ix_selected = tr_rand_ix(1:floor(length(y_te)*0.9) );
        %X_tr = X_tr(ix_selected, :);
        %y_tr = y_tr(ix_selected);

        %if length(y_tr) > 20 && length(y_te) > 20
        [y_pred, auc] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
        dec_mat(r, c) = auc;
        %end
    end
end
fprintf('done using larger clusters to predict smaller clusters\n');
catch exception
end
    
%% --------------------------------------------------------------------
% Using smaller clusters to predict larger clusters (e.g.: B predicts A)
try
for r = 2:n_clusters
    for c = 1 : r - 1
        %r,c,
        %if r == 2
        %    Xb = X([clus{r}; clus{3}], :);
        %    yb = y([clus{r}; clus{3}]);
        %else
            Xb = X(clus{r}, :);
            yb = y(clus{r});
        %end

        Xa = X(clus{c}, :);
        ya = y(clus{c});
        % Using the smaller cluster to train a model to predict the
        %  larger cluster will perform bad for sure, thus make our
        %  evaluation worthless - once the cluster result is
        %  un-balanced, the cross-cluster-validation result will be
        %  low, while when the clusters are balanced, the result would
        %  be high, thus it can not tell whether the data has strong
        %  clusters. 
        % We decide to add extra instances from the bigger cluster to
        %  the smaller cluster to let the new cluster (smaller cluster 
        %  plus instances from bigger cluster) to have the same number
        %  of instances as the bigger cluster. 
        % But I think it may lead to awkward result. For highly
        %  un-balanced clusters, the smaller cluster's 10-fold CV
        %  result might be lower than using the new cluster(small + big)
        %  to predict the bigger cluster. Or maybe the (small + big)
        %  cluster's prediction result for the bigger cluster would be
        %  very close to the bigger cluster's 10-fold CV result, that
        %  means, the more the clustering result is un-balanced, the
        %  lower the final score, while the more the data is balanced,
        %  the more tendence to have a higher final score. It is always
        %  biased.
        %
        % Monday, May 15, 2017
        % Revised according to Dr. Ruan's instruction.
        % From observation of the 3-clusters clustering data, we found
        %  that the clusters are more or less balanced, especially when
        %  we set the kmeans 'Distance' option to 'cosine'. Thus, in
        %  most of cases, we can use a subset of B and C to predict A
        dec_mat(r, c) = nan;
        %if length(yb) > 20
            %if length(yb) < length(ya)
                %[y_pred, auc] = f_weka_RF_arff_10_fo_2(Xa, ya, Xb, yb);
            %else
                % 7-16-2017
                % since in a real applications when people want better 
                %  clustering result, such as more than 3 clusters in some
                %  cases, in those cases, we assume that clusters B, C, D,
                %  E, etc has real meanings so we will simply use the
                %  cluster itself to predict. (won't combine B and C etc.) 
                [y_pred, auc] = f_weka_RF_tr_te(Xb, yb, Xa, ya);
            %end

            dec_mat(r, c) = auc;
        %end
    end
end

fprintf('done using smaller clusters to predict larger clusters\n');
catch exception   
end
    

%% use the best 'other' to predict current cluster for the whole data
y_pred_other = nan(length(y), 1);

try
for j=1:n_clusters
    dec_1_clust = dec_mat(1:end-1, j);
    % 'remove' self
    dec_1_clust(j) = -inf;
    % find the best 'other' cluster for the current cluster.
    ix = find(dec_1_clust == max(dec_1_clust));
    % cluster 'ix' is the best 'other' cluster
    
    X_te = X(clus{j}, :);
    y_te = y(clus{j});    
    X_tr = X(clus{ix}, :);
    y_tr = y(clus{ix});
    [y_pred, ~] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
    
    % norm
    templete = y_pred_whole(clus{j});
    target = y_pred;
    y_pred = f_my_quantileNorm(templete, target);
    % put the normed prediction to the whole data.
    y_pred_other(clus{j}) = y_pred;
end
catch exception
end

y_tmp = y(~isnan(y_pred_other)); y_pred_other = y_pred_other(~isnan(y_pred_other));
auc_other = f_SampleError(y_pred_other, y_tmp, 'AUC');

%% tactically use the best for each cluster
y_pred_tactic = y_pred_whole;

tactic_pick = cell(1, n_clusters); % whole / self / other_x
tactic_pick_code = zeros(1, n_clusters); % 1 for whole; 2 for self; 3 for other

try
for c = 1:n_clusters
    tmp = dec_mat(:, c);
    % ------------- convert nan to 1 --------------
    % Because usually result is nan, it is because y is all 1 or all 0, and
    %  the predictions are all correct. Thus we put all nan to 1. 
    % In addition, we should choose 'whole' when there is a 'draw'.
    %  
    % tmp(isnan(tmp)) = 1;
    % revised from the above line of code. If the cluster's result is nan (
    %  y is all 1 or all 0 and the prediction might be all correct), and 
    %  there is enough instances, we convert the nan to 1 to show it is a
    %  good prediction.
    for q = 1:n_clusters
        if isnan(tmp(q))
            if length(clus{q} ) > 15
                tmp(q) = 1;
            end
        end
    end
    
    % give whole data 0.05 AUC benefit
    % give self data 0.02 AUC benefit
    tmp(end) = tmp(end) + adv_whole;
    tmp(c) = tmp(c) + adv_self;
    if max(tmp) == tmp(4)
        % when whole is best
        y_pred_tactic(clus{c}) = y_pred_whole(clus{c});
        %y_pred_tactic_pcc(clus{c}) = y_pred_whole(clus{c});
        
        tactic_pick{c} = 'whole';
        tactic_pick_code(c) = 1;
    elseif max(tmp) == tmp(c)
        % when self is best 
        y_pred_tactic(clus{c}) = y_pred_self(clus{c});
        %y_pred_tactic_pcc(clus{c}) = y_pred_pcc(clus{c});
        
        tactic_pick{c} = 'self';
        tactic_pick_code(c) = 2;
    else
        % when other is the best
        y_pred_tactic(clus{c}) = y_pred_other(clus{c});
        %y_pred_tactic_pcc(clus{c}) = y_pred_other(clus{c});
        
        tactic_pick{c} = ['other_', int2str(find(tmp == max(tmp) ))];
        tactic_pick_code(c) = 3;
    end
    
end
catch exception
end

y_tmp = y(~isnan(y_pred_tactic)); 
y_pred_tactic_p = y_pred_tactic(~isnan(y_pred_tactic));
auc_tactic = f_SampleError(y_pred_tactic_p, y_tmp, 'AUC');

% y_tmp = y(~isnan(y_pred_tactic_pcc)); 
% y_pred_tactic_pcc = y_pred_tactic_pcc(~isnan(y_pred_tactic_pcc));
% auc_tactic_pcc = f_SampleError(y_pred_tactic_pcc, y_tmp, 'AUC');





end










