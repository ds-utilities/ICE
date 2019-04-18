function [ Y_predicteds ] = f_pred_for_a_layer_clust_NB( train_sets, test_sets, ...
    fold, X, y, idx, models,  auc_ct, groups )
% 
% Xs, ys, 

train_loc = train_sets{fold, 1};
% note that here it uses leave-one-out cross-validation. 
test_loc = test_sets{fold, 1};
% ------------------------ Cluster number ---------------------------
clust = idx(test_loc);

train = f_loc_to_tf(train_loc, length(y));
test = f_loc_to_tf(test_loc, length(y));
% using the cluster information to select from the training instances. I
%  also corresponds to the counting-table models.
train = train & (idx==clust);

X_train = X(train, :);
y_train = y(train, :);
X_test = X(test, :);
y_test = y(test, :);


% c_p = c_p_ori;
% c_n = c_n_ori;
% cnt_y_p = cnt_y_p_ori;
% cnt_y_n = cnt_y_n_ori;
c_p = models{clust, 1};
c_n = models{clust, 2};
cnt_y_p = models{clust, 3};
cnt_y_n = models{clust, 4};

% ---------------------- Adjust the counting tables: ----------------------
if y_test
    for j=1:length(X_test)
        %size(X_test)
        %size(c_p)
        %j,
        %X_test(j)
        c_p(X_test(j), j) = c_p(X_test(j), j) - 1;
    end
else
    for j=1:length(X_test)
        c_n(X_test(j), j) = c_n(X_test(j), j) - 1;
    end
    %c_n(idx) = c_n(idx) -1;
end
% sum(c_p(:) <= 0),

% ---------------------- Adjust y ----------------------
cnt_y_p = cnt_y_p -  y_test;
cnt_y_n = cnt_y_n - ~y_test;
% OK, the result is correct.                    

% % total number of instances
% ni = cnt_y_p + cnt_y_n;

% ------------------------ Evaluation -----------------------------
[Y_predicteds]=f_main_MFG_MF_NF_NB_4(X_train, y_train, X_test, ...
    auc_ct, groups, c_p, c_n, cnt_y_p, cnt_y_n );


end

