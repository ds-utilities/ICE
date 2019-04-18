function [Y_predicteds] = f_pred_MFG_nb_NB(X, groups, ...
     c_p, c_n, cnt_y_p, cnt_y_n,  sim,  tfs, NN )
%  For Naive Bayes Method
% sim: is the similarity array between the test instances and the training
%   instances. By using sim, we can select the good classifiers among all
%   the trained classifiers. The main function is to select the neighbours
%   of each test instance. 
%  Each row correspond to a training instance, each column corresponds to a
%   test instance.
%  Usually, there are 3000 rows represent 3000 positive instances when
%   onlyPosi=1.
% 
% tfs: a logical array with size n_instance by m_classifiers. Each 
%  element indicates if each classifier will be kept or not for the 
%  instance. 1(true) for keep, 0(false) for discard. Each row corresponds
%  to a training instance, each column corresponds to a feature group.
%
% NN: n_neighb, number of neighbours of each instance.
%
% Workflow:
% 1. for each X slice, predict Y_predicteds_slice.
% 2. then combine the predicted y into one vector. 




% ---------------------------- prediction -----------------------------
% convert the tfs training matrix to include the neighbour information
tfs_cnt_test_nb = nan(size(X,1), size(tfs,2));
% tfs_test_nb = zeros(size(tfs));
for i=1:size(X,1)
    sim_col = sim(:, i);
    [~, s_id] = sort(sim_col, 'descend');
    top_neighbs = s_id(1: NN);
    % tfs_test_nb(i, :) = sum( tfs(top_neighbs, :), 1 ) > 0;
    tfs_cnt_test_nb(i, :) = sum( tfs(top_neighbs, :), 1 );
    
end
% fprintf('numberof models used: %d \n', sum(tfs_cnt_test_nb) );
tfs_test_nb = tfs_cnt_test_nb > 0;

% predict for each X_slice
% y_tmp: stores all the y predictions.
y_tmp = zeros(size(X,1), length(groups));
for i=1:length(groups)
    Y_one_slice = nan(size(X, 1), 1);
    % find the instances associated with the current model.
    ix_tf = tfs_test_nb(:,i) ;
    % get the corresponding xs
    
    xs = X(:, groups{i});
    %ix_tf,
    if ix_tf
        xs = xs(ix_tf, :);
        [y_pred_sub, ~] = f_predict_self_NB(c_p(:, groups{i}), c_n(:, groups{i}),...
            cnt_y_p,  cnt_y_n,   xs);
        Y_one_slice(ix_tf) = y_pred_sub;
    else
        %Y_one_slice = nan;
    end        
    y_tmp(:, i) = Y_one_slice;
end

Y_predicteds = y_tmp .* tfs_cnt_test_nb;



% Y_predicteds = nansum(Y_predicteds, 2) ./ (NN.*sum(tfs(1,:)));
%  I think I may made a mistake in the above command.
Y_predicteds = nansum(Y_predicteds, 2) ./ sum(tfs_cnt_test_nb, 2);

% Give the nan values in the prediction array to 1 or 0.
Y_predicteds(isnan(Y_predicteds)) = round(rand([sum(isnan(Y_predicteds)), 1]));


end