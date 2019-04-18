function [auc_whole, auc_self, auc_other, auc_tactic, auc_tactic_pcc, auc_pcc,...
    y_pred_whole, y_pred_self, y_pred_other, y_pred_tactic, ...
    tactic_pick, tactic_pick_code] = ...
    f_tactic_auc_by_clusters_3(X, y, result, adv_whole, adv_self, y_pred_pcc)
% based on v2, this version introduces result of auc_tactic_pcc

% revised from f_tactic_auc_by_clusters(). This version won't re-do the
%  prediction computation again, it will use the already predicted whole,
%  self, other result, and the clustering information to compute the
%  prediction result for tactic and tactic_PCC

clus = result{1, 13};
n_clusters = length(clus);
dec_mat = result{1, 12};

y_pred_whole = result{1, 6};
auc_whole = result{1, 2};

y_pred_self = result{1, 7};
auc_self = result{1, 3};

y_pred_other = result{1, 8};
auc_other = result{1, 4};


%% use whole to predict (single model)
%% use self to predict.
%% use other to predict current cluster
%% tactically use the best for each cluster
y_pred_tactic = y_pred_whole;
y_pred_tactic_pcc = y_pred_whole;

tactic_pick = cell(1, 3); % whole / self / other_x
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
    
    %tmp(4) = tmp(4) + 0.05;
    %tmp(c) = tmp(c) + 0.02;
    tmp(4) = tmp(4) + adv_whole;
    tmp(c) = tmp(c) + adv_self;
    
    if max(tmp) == tmp(4)
        % when whole is best
        y_pred_tactic(clus{c}) = y_pred_whole(clus{c});
        y_pred_tactic_pcc(clus{c}) = y_pred_whole(clus{c});
        
        tactic_pick{c} = 'whole';
        tactic_pick_code(c) = 1;
    elseif max(tmp) == tmp(c)
        % when self is best 
        y_pred_tactic(clus{c}) = y_pred_self(clus{c});
        y_pred_tactic_pcc(clus{c}) = y_pred_pcc(clus{c});
        
        tactic_pick{c} = 'self';
        tactic_pick_code(c) = 2;
    else
        % when other is the best
        y_pred_tactic(clus{c}) = y_pred_other(clus{c});
        y_pred_tactic_pcc(clus{c}) = y_pred_other(clus{c});
        
        tactic_pick{c} = ['other_', int2str(find(tmp == max(tmp) ))];
        tactic_pick_code(c) = 3;
    end
    
end
catch exception
end

y_tmp = y(~isnan(y_pred_tactic)); 
y_pred_tactic = y_pred_tactic(~isnan(y_pred_tactic));
auc_tactic = f_SampleError(y_pred_tactic, y_tmp, 'AUC');

y_tmp = y(~isnan(y_pred_tactic_pcc)); 
y_pred_tactic_pcc = y_pred_tactic_pcc(~isnan(y_pred_tactic_pcc));
auc_tactic_pcc = f_SampleError(y_pred_tactic_pcc, y_tmp, 'AUC');

auc_pcc = f_SampleError(y_pred_pcc, y_tmp, 'AUC');


end





