function [c_p, c_n, cnt_y_p, cnt_y_n ] = f_cnt_freq_NB(X, y)
% For each feature, for each value of y (class label), count the frequency 
%   for each value. 
% Note: here y can be 1 or 0.
%
% c_p: counts for positive instances, for each feature.
%  each column corresponds to a feature
%  each row corresponds to a category of X (possible value of X)


% uniqueVs = unique(X(:));
uniqueVs = 1:5;

c_p = zeros(length(uniqueVs), size(X,2));
c_n = zeros(length(uniqueVs), size(X,2));
for i=1:size(X,2)
    [cnts_p, cnts_n] = f_cnt_freq_NB_col(X(:,i), y, uniqueVs);
    c_p(:, i) = cnts_p;
    c_n(:, i) = cnts_n;
    
end

cnt_y_p = sum(y==1);
cnt_y_n = sum(y==0);

% Laplace Smooth:
c_p = c_p + 1./length(uniqueVs);
c_n = c_n + 1./length(uniqueVs);

end

