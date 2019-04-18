function [train, test] = f_Kfold_cv(y, k_fold)
% Alias of f_Kfold_cv_matlab
%  wrong: Alias of f_StratifiedKFold.
% form the sets for cross-validation.
% n: length of the data

% I found that my version of k fold is wrong, that each fold will have very
% un-stable number of 0 or 1. But matlab version is correct.
% [train, test] = f_StratifiedKFold(n, k_fold);
[train, test] = f_Kfold_cv_matlab(y, k_fold);



end

