function [y_pred_tactic, auc_tactic, tactic_pick, tactic_pick_code] = ...
    f_tune_use_AB_dec_tab_3(X, y, clus, adv_whole, adv_self, ...
    dec_mat, y_pred_whole, y_pred_self, y_pred_other)


% Revised from f_AB_dec_tab_3(). Tune adv_whole and adv_self on a data

n_clusters = length(clus);

%% --------------------------- WHOLE -------------------------------------
% Predict each cluster using the whole data. 
%% ----------------- SELF ---------------------------------------
% Self prediction using 10-fold CV
%% ---------------------- Find best 'other' ------------------------------
% fill dec_mat for 'others'
% Using larger clusters to predict smaller clusters (e.g.: A predicts B)
%% --------------------------------------------------------------------
% Using smaller clusters to predict larger clusters (e.g.: B predicts A)
%% use the best 'other' to predict current cluster for the whole data
%% tactically use the best for each cluster
y_pred_tactic = y_pred_whole;

tactic_pick = cell(1, n_clusters); % whole / self / other_x
tactic_pick_code = zeros(1, n_clusters); % 1 for whole; 2 for self; 3 for other

% try
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
    if max(tmp) == tmp(end)
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
        
        % tactic_pick{c} = ['other_', int2str(find(tmp == max(tmp) ))];
        tmp_ixs = find(tmp == max(tmp));
        first_ixs = tmp_ixs(1);
        tactic_pick{c} = ['other_', int2str(first_ixs )];
        tactic_pick_code(c) = 3;
    end
    
end
% catch exception
% end

y_tmp = y(~isnan(y_pred_tactic));

y_pred_tactic_p = y_pred_tactic(~isnan(y_pred_tactic));
auc_tactic = f_SampleError(y_pred_tactic_p, y_tmp, 'AUC');

% y_tmp = y(~isnan(y_pred_tactic_pcc)); 
% y_pred_tactic_pcc = y_pred_tactic_pcc(~isnan(y_pred_tactic_pcc));
% auc_tactic_pcc = f_SampleError(y_pred_tactic_pcc, y_tmp, 'AUC');



fprintf('done tactically use the best for each cluster\n');


end










