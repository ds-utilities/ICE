function [auc_whole, auc_self, auc_other, auc_tactic,...
    y_pred_whole, y_pred_self, y_pred_other, y_pred_tactic, ...
    tactic_pick, tactic_pick_code] = ...
    f_tactic_auc_by_clusters(X, y, clus, n_clusters, alg, dec_mat)
% 

%% use whole to predict (single model)
[y_pred_whole, auc_whole] = f_weka_RF_arff_10_fo_3(X, y);

% 
%% use self to predict.
y_pred_self = nan(length(y), 1);
% Self prediction using 10-fold CV
try
for j=1:n_clusters
    Xj = X(clus{j}, :);
    yj = y(clus{j});

%     if length(yj) > 20
    if length(yj) > 15
        [y_pred, ~] = f_weka_RF_arff_10_fo_3(Xj, yj);
        
        y_pred_self(clus{j}) = y_pred;
    end
end
fprintf('done self evaluation\n');
catch exception
end

y_tmp = y(~isnan(y_pred_self)); y_pred_self = y_pred_self(~isnan(y_pred_self));
auc_self = f_SampleError(y_pred_self, y_tmp, 'AUC');

%% use other to predict current cluster
y_pred_other = nan(length(y), 1);
% ----------------- for cluster A (B or B+C predicts A) -----------------
try
r = 2; % using B to predict A
c = 1;
%r,c,
    Xb = X([clus{r}; clus{3}], :);
    yb = y([clus{r}; clus{3}]);
    
    Xa = X(clus{c}, :);
    ya = y(clus{c});

    %if length(yb) > 20
        %if length(yb) < length(ya)
            %[y_pred, auc] = f_weka_RF_arff_10_fo_2(Xa, ya, Xb, yb);
        %else
            [y_pred, auc] = f_weka_RF_tr_te(Xb, yb, Xa, ya);
        %end

        %dec_mat(r, c) = auc;
    %end
    y_pred_other(clus{1}) = y_pred;
catch exception   
end
% --------------- for cluster B and C (A predicts B and C) ----------------
try
    %r = 1;
    for c = 2:3
        tmp = dec_mat(:, c);
        if c == 2
            tmp = tmp([1,3]);
            if tmp(1) >= tmp(2)
                r = 1;
            else
                r = 3;
            end
        elseif c == 3
            tmp = tmp([1,2]);
            if tmp(1) >= tmp(2)
                r = 1;
            else
                r = 2;
            end            
        end
        
        X_tr = X(clus{r}, :);
        y_tr = y(clus{r});            
        X_te = X(clus{c}, :);
        y_te = y(clus{c});
        %dec_mat(r, c) = nan;

        % from training instances, random pick up instances equals to 
        %  90% test instances size for a fair comparison to 10-fold
        %  CV on testing data.
        %tr_rand_ix = randperm(length(y_tr));
        %ix_selected = tr_rand_ix(1:floor(length(y_te)*0.9) );
        %X_tr = X_tr(ix_selected, :);
        %y_tr = y_tr(ix_selected);
        
        %if length(y_tr) > 20 && length(y_te) > 20
        [y_pred, auc] = f_weka_RF_tr_te(X_tr, y_tr, X_te, y_te);
        %dec_mat(r, c) = auc;
        %end

        y_pred_other(clus{c}) = y_pred;
    end
catch exception
end

y_tmp = y(~isnan(y_pred_other)); 
y_pred_other = y_pred_other(~isnan(y_pred_other));
auc_other = f_SampleError(y_pred_other, y_tmp, 'AUC');


%% tactically use the best for each cluster
y_pred_tactic = y_pred_whole;

tactic_pick = cell(1, 3);
tactic_pick_code = zeros(1, 3); % 1 for whole; 2 for self; 3 for other
try
for c = 1:n_clusters
    tmp = dec_mat(:, c);
    
    % ------------- convert nan to 1 --------------
    % because usually result is nan, it is because y is all 1 or all 0, and
    %  the predictions are all correct. And we should choose 'whole' when
    %  there is a 'draw'.
    tmp(isnan(tmp)) = 1;
    % give whole data 0.05 AUC benefit
    % give self data 0.02 AUC benefit
    % other cluster predictions have no benefit
    tmp(4) = tmp(4) + 0.05;
    tmp(c) = tmp(c) + 0.02;
    
    if max(tmp) == tmp(4)
        % when whole is best
        y_pred_tactic(clus{c}) = y_pred_whole(clus{c});
        
        tactic_pick{c} = 'whole';
        tactic_pick_code(c) = 1;
    elseif max(tmp) == tmp(c)
        % when self is best 
        y_pred_tactic(clus{c}) = y_pred_self(clus{c});
        
        tactic_pick{c} = 'self';
        tactic_pick_code(c) = 2;
    else
        % when other is the best
        y_pred_tactic(clus{c}) = y_pred_other(clus{c});
        
        tactic_pick{c} = ['other_', int2str(find(tmp == max(tmp) ))];
        tactic_pick_code(c) = 3;
    end
    
end
catch exception
end

y_tmp = y(~isnan(y_pred_tactic)); 
y_pred_tactic = y_pred_tactic(~isnan(y_pred_tactic));
auc_tactic = f_SampleError(y_pred_tactic, y_tmp, 'AUC');


end

