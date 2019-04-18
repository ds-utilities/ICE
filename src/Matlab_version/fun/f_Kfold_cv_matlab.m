function [train, test] = f_Kfold_cv_matlab(y, k_fold)
% Matlab realization of stratified k-fold CV

c = cvpartition(y, 'k', k_fold);
train = {};
test = {};
for i=1:k_fold
    train{i, 1} = find(c.training(i));
    test{i, 1} = find(c.test(i));
end


end