function groups = f_gen_grps(n, r, nRP)
% Generate groups by random permutation and divide into r groups, and
%  repeat this nRP times
% n: Number of columns (features) of the data set. 
%  n = size(X, 2); % if X is given.
% r: number of groups
% nRP: number of random permutations.

groups = [];
for i =1:nRP
    fprintf('Re-sampling... current re-sampling id: %d \n', i);
    
    g = f_randPerm_r_grps(n, r);
    groups = [groups; g];
end

end

