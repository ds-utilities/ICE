function groups = f_randPerm_r_grps(n, r)
% n: Number of columns (features) of the data set
% r: Divide into r groups 

p = randperm(n);
% number of attributes each group
a = ceil(n/r);

% r is the rough number of groups, w is the real number of groups
w = ceil(n/a);
groups = {}; % group: size: w by 1. w is the number of groups.
% groups = cell(w, 1);

for i=1:w-1
    groups{i,1} = p(a*(i-1) + 1: a*(i-1)+a);
end
groups{w,1} = p(a*(w-1) + 1: end);



end

