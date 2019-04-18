function [y_pred] = ...
    f_stack_slash_pred_one(preds, nb_class_labels,nb_self_preds, n_model)
%
% preds: 1*900 predictions by 900 models from 180 neighbors
% nb_class_labels: 1*180; 180 neighbors' class labels
% nb_self_preds: neighbors self predictions. 1*900
% nb_tfs: 1*900, model availabilities.

% number of neighborhoods
n_nbhoods = length(preds) / length(nb_class_labels);

%% for the 1*900 arrays, reshape them to 180*5 arrays
preds_1 = reshape(preds, [n_nbhoods, length(nb_class_labels)])';
nb_self_preds_1 = reshape(nb_self_preds, [n_nbhoods, length(nb_class_labels)])';
% nb_tfs_1 = reshape(nb_tfs, [n_nbhoods, length(nb_class_labels)])';

nb_class_labels_1 = nb_class_labels';

%% error of the neighbor's self predictions
err = abs(repmat(nb_class_labels_1, [1, n_nbhoods]) - nb_self_preds_1) ;

% sort the error array for each row
[~, ixs]=sort(err, 2);

% for the n_neighbors * 5 y_pred data, give the neighbor's class label to
%  those with wrong models
[preds_1] = f_give_label_to_err_model(preds_1, nb_class_labels_1);

%% sort the preds_1 array by the models' error
sorted_preds_1 = zeros(size(preds_1));
for i=1:size(preds_1, 1)
    row = preds_1(i, :);
    sorted_preds_1(i, :) = row(ixs(i, :));
end

%% use the slash triangle selection method to select preds for final result
% only use 3 models for each inst.
% n_top_mod_each_nb = 3;
n_top_mod_each_nb = 3;
rank = f_slash_triangle_rank(zeros(size(preds_1, 1), n_top_mod_each_nb)); 
rank = [rank, zeros(size(preds_1, 1),  n_nbhoods-n_top_mod_each_nb)];

y_preds = zeros(n_model, 1);
for i=1:n_model
    ix = rank==i;
    %size(sorted_preds_1(ix)),
    if numel(sorted_preds_1(ix))==1
        y_preds(i) = sorted_preds_1(ix);
    else
        y_preds(i) = nan;
    end
end

y_pred = nanmean(y_preds);



end

