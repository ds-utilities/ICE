function [X_sets, groups] = f_divide_attri_rand_perm(X, r)
% Gives a data set X, with size m by n, divide X to several columns by
%  random select the attributes. 
% r is the number of attribute groups 

% if nargin < 2
%     % number of groups
%     r = 10;
% end

% n: Number of columns (features) of the data set
n = size(X, 2);

% ----------------------- Divide into r groups  ---------------------------
groups = f_randPerm_r_grps(n, r);

% ------------------------ divide X by groups -----------------------------
X_sets = f_groups2X_sets(X, groups);
    % X_sets = {};
    % % X_sets = cell(1, w);
    % for i=1:w
    %     X_sets{1, i} = X(:, groups{i});
    % end

end

