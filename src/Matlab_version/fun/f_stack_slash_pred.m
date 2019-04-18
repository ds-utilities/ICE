function [y_preds] = ...
    f_stack_slash_pred(predss, nb_class_labels,nb_self_predss, n_model)
%
y_preds = zeros(size(predss, 1), 1);
for i=1:size(predss, 1)
    y_preds(i)=f_stack_slash_pred_one(predss(i,:), nb_class_labels(i,:),...
    nb_self_predss(i,:), n_model);
end




end

