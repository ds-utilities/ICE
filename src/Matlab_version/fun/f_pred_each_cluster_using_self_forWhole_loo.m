function [dec_mat, y_pred_self, auc_self, y_pred_self_multi] = ...
    f_pred_each_cluster_using_self_forWhole_loo(X, y, clus, dec_mat, n_clusters,...
    y_pred_whole, useParfor)
% Self prediction using 10-fold CV

if nargin < 7
    useParfor = 0;
end

% y_pred_self = nan(length(y), 1);
% also save the result for instances that belongs to more than one
%  clusters.
y_pred_self_multi = nan(length(y), n_clusters);



% try
for j=1:n_clusters
    Xj = X(clus{j}, :) ;
    yj = y(clus{j}) ;
    dec_mat(j, j) = nan;

    if length(yj) > 15
        if useParfor == 1
            [y_pred, auc] = f_weka_RF_arff_k_fo_3_parfor(Xj, yj, length(yj));
        else
            [y_pred, auc] = f_weka_RF_arff_k_fo_3(Xj, yj, length(yj));
        end
        
        dec_mat(j, j) = auc;
        
        % Normalization
        templete = y_pred_whole(clus{j});
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);
        
        % copy the normed prediction to the whole data.
        %y_pred_self(clus{j}) = y_pred;
        y_pred_self_multi(clus{j}, j) = y_pred;

    end
end
fprintf('done self evaluation\n');
% catch exception
% end

% convert the n_inst * n_cluster y_pred array to one column.
y_pred_self = nanmean(y_pred_self_multi, 2);

y_tmp = y(~isnan(y_pred_self)); 
y_pred_self_noNan = y_pred_self(~isnan(y_pred_self));
auc_self = f_SampleError(y_pred_self_noNan, y_tmp, 'AUC');

% replace the nan values into average y_pred.
y_pred_self(isnan(y_pred_self)) = nanmean(y_pred_self);



end

