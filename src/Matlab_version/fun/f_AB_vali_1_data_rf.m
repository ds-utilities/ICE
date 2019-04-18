function [val_mat, val_mat_kap] = f_AB_vali_1_data_rf(X, y, clus, n_clusters)
% 

% val_mat: validation matrix. 3+1 rows and 3 columns. Take the first 
%  row for example. a(1,1) is the 10-fold cv AUC on cluster A; a(1,2) 
%  is the AUC of A predict B; a(1,3) is the AUC of A predict C.
% The last row is using the whole data to train a model and to predict
%  all A, B and C clusters. 
val_mat = nan(n_clusters + 1, n_clusters);
val_mat_kap = nan(n_clusters + 1, n_clusters);
    %% --------------------------------------------------------------------
    % Self prediction using 10-fold CV
    try
    for j=1:n_clusters
        Xj = X(clus{j}, :);
        yj = y(clus{j});
        val_mat(j, j) = nan;
        
        if length(yj) > 20
            %[y_pred, auc, kap] = f_weka_RF_arff_10_fo(Xj, yj);
            %[y_pred, auc, kap] = f_weka_svm_arff_k_fo_3_parfor(Xj, yj, 10);
            [y_pred, auc, kap] = f_weka_RF_arff_k_fo_3_parfor(Xj, yj, 10, 100);
            
            val_mat(j, j) = auc;
            val_mat_kap(j, j) = kap;
        end
    end
    fprintf('done self evaluation\n');
    catch exception
    end
    
    %% --------------------------------------------------------------------
    % Using larger clusters to predict smaller clusters
    
    try
    for r = 1:n_clusters
        for c = r + 1 : n_clusters
            X_tr = X(clus{r}, :);
            y_tr = y(clus{r});            
            X_te = X(clus{c}, :);
            y_te = y(clus{c});
            val_mat(r, c) = nan;
            
            % from training instances, random pick up instances equals to 
            %  90% test instances size for a fair comparison to 10-fold
            %  CV on testing data.
            tr_rand_ix = randperm(length(y_tr));
            ix_selected = tr_rand_ix(1:floor(length(y_te)*0.9) );
            X_tr = X_tr(ix_selected, :);
            y_tr = y_tr(ix_selected);
            
            %if length(y_tr) > 20 && length(y_te) > 20
            %[y_pred, auc, kap] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
            %[y_pred, auc, kap] = f_weka_svm2_tr_te(X_tr, y_tr, X_te, y_te);
            [y_pred, auc, kap] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te, 100);
            
            val_mat(r, c) = auc;
            val_mat_kap(r, c) = kap;
            %end
        end
    end
    fprintf('done using larger clusters to predict smaller clusters\n');
    catch exception
    end
    
    %% --------------------------------------------------------------------
    % Using smaller clusters to predict larger clusters
    try
    for r = 2:n_clusters
        for c = 1 : r - 1
            %r,c,
            if r == 2
                Xb = X([clus{r}; clus{3}], :);
                yb = y([clus{r}; clus{3}]);
            else
                Xb = X(clus{r}, :);
                yb = y(clus{r});
            end
            
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
            val_mat(r, c) = nan;
            if length(yb) > 20
                if length(yb) < length(ya)
                    %[y_pred, auc, kap] = f_weka_svm_arff_10_fo_2(Xa, ya,Xb, yb);
                    [y_pred, auc, kap] = f_weka_RF_arff_10_fo_2(Xa, ya,Xb, yb);
                    
                else
                    %[y_pred, auc, kap] = f_weka_svm2_tr_te(Xb, yb, Xa, ya);
                    [y_pred, auc, kap] = f_weka_RF_tr_te(Xb, yb, Xa, ya, 100);
                    
                end
                
                val_mat(r, c) = auc;
                val_mat_kap(r, c) = kap;
            end
        end
    end
    
    fprintf('done using smaller clusters to predict larger clusters\n');
    catch exception   
    end
    
    %% --------------------------------------------------------------------
    % Predict each cluster using the whole data. 
    try
    %[y_pred, ~] = f_weka_RF_arff_10_fo_3(X, y);
    %[y_pred, ~] = f_weka_svm_arff_k_fo_3_parfor(X, y);
    [y_pred, ~] = f_weka_RF_arff_k_fo_3_parfor(X, y, 10);
    
    for j=1:n_clusters
        
        auc = f_SampleError(y_pred(clus{j}), y(clus{j}), 'AUC');
        kap = f_my_01_kappa(y_pred(clus{j}), y(clus{j}) );
        val_mat(n_clusters+1, j) = auc;
        val_mat_kap(n_clusters+1, j) = kap;
    end
        
    fprintf('done using whole data to predict each clusters\n');
    catch exception   
    end
    


end

