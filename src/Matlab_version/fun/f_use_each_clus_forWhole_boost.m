function [y_pred_multi] = ...
    f_use_each_clus_forWhole_boost(X, y, clus, y_pred_whole, useParfor, I)
% predict the whole instances using each cluster data, while self 
%  prediction using 10-fold CV.

if nargin < 5
    useParfor = 1;
    I = 100;
end
n_clusters = length(clus);
% y_pred_self = nan(length(y), 1);
% also save the result for instances that belongs to more than one
%  clusters.
y_pred_multi = nan(length(y), n_clusters);



% try
for j=1:n_clusters
    Xj = X(clus{j}, :) ;
    yj = y(clus{j}) ;

    fprintf('Cluster %d started...\n', j);
    if length(yj) > 15
        % ------------------ for self ------------------
        if useParfor == 1
            [y_pred, auc, ~] = f_weka_boost_arff_k_fo_3_parfor(Xj, yj, 10, I);
        else
            [y_pred, auc, ~] = f_weka_boost_arff_k_fo_3(Xj, yj, 10, I);
        end        
        % Normalization
        templete = y_pred_whole(clus{j});
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);
        
        % copy the normed prediction to the whole data.
        %y_pred_self(clus{j}) = y_pred;
        y_pred_multi(clus{j}, j) = y_pred;

        fprintf('Cluster %d done self evaluation\n', j);
        
        % ------------------ for other ------------------
        %ix_other = [];
        %for q = 1:n_clusters
        %    if q ~= j
        %        ix_other = [ ix_other, clus{q}];
        %    end
        %end
        ix_other = setdiff( (1:length(y))', clus{j} );
        
        X_other = X( ix_other , :) ;
        y_other = y( ix_other ) ;
        % weka prediction
        y_pred = f_weka_boost_tr_te(Xj, yj, X_other, y_other, I);
        % Normalization
        templete = y_pred_whole(ix_other);
        target = y_pred;
        y_pred = f_my_quantileNorm(templete, target);
        % copy to the whole array
        y_pred_multi(ix_other, j) = y_pred;
        fprintf('Cluster %d done predicting other instances\n', j);
        
        
    end
    
    
    
end
% catch exception
% end



end

