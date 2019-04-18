function X_sets = f_groups2X_sets(X, groups)
% subsets of f_divide_attri_rand_perm()
% input groups and X, return X_sets

% divide X
X_sets = {};
% X_sets = cell(1, w);
for i=1:length(groups)
    X_sets{1, i} = X(:, groups{i});
end

end