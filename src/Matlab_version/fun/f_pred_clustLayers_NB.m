function [ Y_predicted ] = f_pred_clustLayers_NB( train_sets, test_sets, ...
    fold, X, y, idxs, modelss,  auc_ct, groups, layers)

% total_layes = size(idxs, 2);

preds = [];
for layer = layers
    % layer,
    % size(modelss)
    [ Y_predicteds ] = f_pred_for_a_layer_clust_NB( train_sets, test_sets, ...
        fold, X, y, idxs(:, layer), modelss{1, layer},  auc_ct, groups );
    preds = [preds, Y_predicteds];
    
end

Y_predicted = nanmean(preds);

end