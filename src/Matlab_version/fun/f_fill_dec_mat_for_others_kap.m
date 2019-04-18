function [dec_mat_kap] = f_fill_dec_mat_for_others_kap(X, y, clus, ...
    dec_mat_kap, n_clusters, y_pred_whole, y_pred_self_multi )
%  
% fill dec_mat for 'others'

%% Using larger clusters to predict smaller clusters (e.g.: A predicts B)

% try
for r = 1:n_clusters % r: for each row - each training cluster
    for c = r + 1 : n_clusters % c: for each col - each testing cluster
        X_tr = X(clus{r}, :);
        y_tr = y(clus{r});            
        X_te = X(clus{c}, :);
        y_te = y(clus{c});
        % dec_mat(r, c) = nan;
        dec_mat_kap(r, c) = nan;
        
        % weka prediction
        y_pred = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
        
        % Normalization
        templete = y_pred_whole(clus{c});
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);
        
        % change the biased predicted instances (the ones also belongs to 
        % the training insts.) to un-biased predictions. Change these to
        % clus{r}'s prediction. (which is un-biased)
        tf_te = false(length(y), 1);
        tf_te(clus{c}) = true;
        tf_tr = false(length(y), 1);
        tf_tr(clus{r}) = true;
        % insts. for both cluster:
        tf_both = tf_te & tf_tr;
        y_pred_corrected = y_pred_self_multi(tf_both, r);
        tf_biased_on_te = tf_both(tf_te);
        
        y_pred(tf_biased_on_te) = y_pred_corrected;
        
        % compute AUC
        %auc = f_SampleError(y_pred, y_te, 'AUC');
        kap = f_my_01_kappa(y_pred, y_te);
        
        %dec_mat(r, c) = auc;
        dec_mat_kap(r, c) = kap;
        %y_te,
        %r,c,auc,
    end
end
fprintf('done using larger clusters to predict smaller clusters\n');
% catch exception
% end
    
%% --------------------------------------------------------------------
% Using smaller clusters to predict larger clusters (e.g.: B predicts A)
% try
for r = 2:n_clusters % r: for each row - each training cluster
    for c = 1 : r - 1 % c: for each col - each testing cluster

        X_tr = X(clus{r}, :);
        y_tr = y(clus{r});
        X_te = X(clus{c}, :);
        y_te = y(clus{c});
        %dec_mat(r, c) = nan;
        dec_mat_kap(r, c) = nan;

        % weka prediction
        y_pred = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);

        % Normalization
        templete = y_pred_whole(clus{c});
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);

        % change the biased predicted instances (the ones also belongs to
        % the training insts.) to un-biased predictions. Change these to
        % clus{r}'s prediction
        tf_te = false(length(y), 1);
        tf_te(clus{c}) = true;
        tf_tr = false(length(y), 1);
        tf_tr(clus{r}) = true;
        % insts. for both cluster:
        tf_both = tf_te & tf_tr;
        y_pred_corrected = y_pred_self_multi(tf_both, r);
        tf_biased_on_te = tf_both(tf_te);

        y_pred(tf_biased_on_te) = y_pred_corrected;

        % compute AUC
        %auc = f_SampleError(y_pred, y_te, 'AUC');
        kap = f_my_01_kappa(y_pred, y_te);
        %dec_mat(r, c) = auc;
        dec_mat_kap(r, c) = kap;
        %r,c,auc,
    end
end

fprintf('done using smaller clusters to predict larger clusters\n');
% catch exception
% end
    



end

