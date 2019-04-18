function [preds_1] = f_give_label_to_err_model(preds_1, nb_class_labels_1)
% for the n_neighbors * 5 y_pred data, give the neighbor's class label to
%  those with wrong models

for i=1:size(preds_1, 1)
    for j = 1:size(preds_1, 2)
        if isnan(preds_1(i, j))
            if nb_class_labels_1(i) == 1
                %preds_1(i, j) = 0.7;
                preds_1(i, j) = 0.7;
            else % if label is 0
                %preds_1(i, j) = 0.3;
                preds_1(i, j) = 0.3;
            end
        end
    end
end




end

