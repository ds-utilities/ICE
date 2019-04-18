function [Y_predicteds]=f_main_MFG_MF_NF_NB_4(X_train, y_train, X_test, ...
    auc_ct, groups, c_p, c_n, cnt_y_p, cnt_y_n )
% Modify for the newer algorithm that averaging multi-layers predictions. 
%  The only change is that this version returns a row of y-predicts for a
%  single instance by different top models. (Won't do average inside this 
%  function, do it outside with other layer predictions together.)
% 
% No model-insts association, just pick up the best models with the best
%  AUCs. When we want to do instance specific association, we will use
%  another script that make several layers of clustering, and for each
%  instance has several best models build for a cluster, which makes the
%  instance-specific models. 

% sim: the similarity matrix.
% NN: consider n top neighbours when do prediction
% 

% ----------------------- Train Classifiers -------------------------------
% The Naive Bayes model don't need to train models at this time, the
%  couting table based model are pre-computed.

% ---------------- Find top models for all instances ----------------------
%                   1. predict y for each X_slice
Y_preds = zeros(size(X_train,1), length(groups));
for i=1:length(groups)
    [Y_pred, ~]=f_predict_self_NB(c_p(:, groups{i}), c_n(:, groups{i}), ...
        cnt_y_p, cnt_y_n,  X_train(:, groups{i}));
    Y_preds(:, i) = Y_pred;
end

%            Rank the models for all instances based on AUC
aucs = zeros(length(groups), 1);
for i=1:length(groups)
    aucs(i) = f_SampleError(Y_preds(:, i), y_train, 'AUC');
end
% tfs indicates the top models among all the feature groups.
tfs = aucs > auc_ct ;
% aucs,

% ----------------------- RE-train classifiers ----------------------------
% No re-train

% --------------------------- Predictions ---------------------------------
loc = find(tfs);

% y_tmp: stores all the y predictions from the top models.
y_tmp = zeros(size(X_test,1), length(loc));
for i=1:length(loc)
    % fprintf('loc(i) = %d \n', loc(i));
    
    [Y_one_slice, ~] = f_predict_self_NB(c_p(:, groups{loc(i)}), c_n(:, groups{loc(i)}),...
        cnt_y_p,  cnt_y_n,   X_test(:, groups{loc(i)}));
    y_tmp(:, i) = Y_one_slice;
    
end

% Y_predicteds = nanmean(y_tmp, 2);
Y_predicteds = y_tmp;

end

