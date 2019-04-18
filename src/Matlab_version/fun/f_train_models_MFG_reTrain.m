function mss = f_train_models_MFG_reTrain(X,Y,model_type,...
    tfs_reTrain, groups, mss)
% re-train models
% tfs_reTrain has the size of #instances x #models
for i=1:length(groups)
    if length(groups{i}) >=3 % just to make sure the code is correct, 
        % probably will always run this branch.
        %[ms, clus] = f_train_models_clust(X_sets{i},Y,...
        %    model_type,clust_alg,q,onlyPosi);
        row_index = tfs_reTrain(:, i);
        col_index = groups{i};
        m = f_train_model(X(row_index, col_index), Y(row_index), model_type);
        ms = {m};
        % ms is just one model, clus is just all the instances.
        mss{1, i} = ms;
        %cluss{1, i} = clus;
    else 
        % Dummy, may not ever goes to here, just in case a group only has 
        %  less than 2 instances.
        mss{1, i} = '';
        %cluss{1, i} = '';
    end
    %fprintf('  group %d\n', i);
end




end