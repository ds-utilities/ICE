function [tfs, tfs_retrain]=f_feGrp_Method5(X, Y, model_type, ...
    groups, mss, sm1, sm2)
% use 2 sms for re-train and instance-model association.
%
%  sm1 -> tfs_retrain for re-train. 
%  sm2 is used for instance-models association. 
%
%  sm1, more stringenet, to make each model show more personality; 
%  sm2, less stringent, since results show that more models give better AUC
%
% =================== 1. predict y for each X_slice =======================
Y_preds = zeros(size(X,1), length(groups));
for i=1:length(groups)
    %Y_preds(:, i) = f_predict_clust_self(X(:, groups{i}), Y, ...
    %    mss{i}, cluss{i}, model_type, onlyPosi);
    Y_preds(:, i) = f_predict_self(X(:, groups{i}), mss{i}, model_type);
end

% =========== For each model, find good instances for Re-train ============
% tfs_retrain: to decide which instances to re-train
[tfs_retrain, ~] = f_get_tfs_reTrain(Y_preds, Y, sm1);

[tfs, ~] = f_get_tfs_reTrain(Y_preds, Y, sm2);



% ------ if one X_slice is empty, just delete all the column of tfs -------
for i=1:length(groups)
    if isempty(mss{i})
        tfs(:, i) = false;
    end
end

end








