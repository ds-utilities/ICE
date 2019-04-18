function [Y_predicteds] = f_predict_MFG_nb(X, groups, mss, sim, ...
    model_type, tfs, NN, isWekaJava, useMATLAB)
% FGC means Feature Grouping + Clustering
%
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

% if nargin < 8
%     % number of neighbors to use in the test stage. 
%     NN = 10;
% end
% if nargin < 7
%     tfs = true(size(sim,1), length(groups));
% end
% % Priviously, I made a mistake here by mis-count the nargin position, 
% %  where for all input of NN, it will use NN=10.

% ---------------------------- divide X -------------------------------
X_sets = cell(1, length(groups));
for i=1:length(groups)
    X_sets{1, i} = X(:, groups{i});
end

% ---------------------------- prediction -----------------------------
% convert the tfs training matrix to include the neighbour information
tfs_test_nb = zeros(size(X,1), size(tfs,2));
% tfs_test_nb = zeros(size(tfs));
for i=1:size(X,1)
    sim_col = sim(:, i);
    [~, s_id] = sort(sim_col, 'descend');
    top_neighbs = s_id(1: NN);
    tfs_test_nb(i, :) = sum( tfs(top_neighbs, :), 1 );
    % previously, I made a mistake here, without ', 1' in the above
    %  command, when top_neighbs=1, sum(tf) will not return a row, it will
    %  return a number, which cause wrong result.
end

% predict for each X_slice
% y_tmp: stores all the y predictions.
y_tmp = zeros(size(X,1), length(groups));
for i=1:length(groups)
    Y_one_slice = nan(size(X, 1), 1);
    % find the instances associated with the current model 
    ix_tf = tfs_test_nb(:,i) ~=0 ;
    % get the corresponding xs
    xs = X_sets{1, i};
    xs = xs(ix_tf, :);
    if ~isempty(mss{i})
        y_pred_sub = f_predict_simple(xs, mss{i}{1}, model_type, ...
            isWekaJava, useMATLAB);
        if ~strcmp(model_type, 'linear_regress')
            system(['rm -f ', mss{i}{1}]);
        end
        
    else
        fprintf('model is empty \n');
    end
    Y_one_slice(ix_tf) = y_pred_sub;
    y_tmp(:, i) = Y_one_slice;
end
Y_predicteds = y_tmp .* tfs_test_nb;
% figure,
% subplot(1,3,1); imagesc(y_tmp);
% subplot(1,3,2); imagesc(Y_predicteds);

% Y_predicteds = nansum(Y_predicteds, 2) ./ (NN.*sum(tfs(1,:)));
%  I think I may made a mistake in the above command.
Y_predicteds = nansum(Y_predicteds, 2) ./ sum(tfs_test_nb, 2);

% Give the nan values in the prediction array to 1 or 0.
Y_predicteds(isnan(Y_predicteds)) = round(rand([sum(isnan(Y_predicteds)), 1]));

% figure, 
% subplot(1,2,1); imagesc(tfs);
% subplot(1,2,2); imagesc(tfs_test_nb);

% f_simple_statistics(tfs_test_nb(:)),
% figure, boxplot(tfs_test_nb(:))

end