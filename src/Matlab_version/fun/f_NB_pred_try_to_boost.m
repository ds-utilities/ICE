function [prob_p, prob_n ] = f_NB_pred_try_to_boost(c_p, c_n, cnt_y_p, ...
    cnt_y_n, Xi, yi,  isFast)
% Adjust the counting table and predict the y. This function is used for
%  the leave-one-out corss-validation.

% [Xi, yi] is one instance.
% [c_p, c_n, y,] is the big X counting tables

% here assuming the values of Xi is converted from 
%   [-2, -1, 0, 1, 2] to [1,2,3,4,5]

% isFast: use the fast matrix multiplication for the posterio prob, or, 
%  simply use a for loop. 1 for use fast; 0 for use 'for' loop.
if nargin < 7
    isFast = 0;
end

% ---------------------- Adjust the counting tables: ----------------------
%     % number of total elements of the counting table.
%     total_ele = numel(c_p);
%     % number of features
%     n_fe = length(Xi);
%     n_rows = size(c_p, 1);
%     tmp = 0:n_rows:(total_ele-1);    
%     % idx = (Xi + 0:5:numel(c_n)),
%     idx = (Xi + tmp )';    
if yi
    for j=1:length(Xi)
        c_p(Xi(j), j) = c_p(Xi(j), j) -1;
    end
else
    for j=1:length(Xi)        
        c_n(Xi(j), j) = c_n(Xi(j), j) -1;
    end
    %c_n(idx) = c_n(idx) -1;
end


% ---------------------- Adjust y ----------------------
%     if yi == 1
%         cnt_y_p = cnt_y_p-1;
%     elseif yi == 0
%         cnt_y_n = cnt_y_n-1;
%     end
% boost speed:
% yi needs to be logical rather than number
cnt_y_p = cnt_y_p -  yi;
cnt_y_n = cnt_y_n - ~yi;
% OK, the result is correct.


% total number of instances
ni = cnt_y_p + cnt_y_n;

% ---------------------- Compute the posterior prob ----------------------
% prob_p = (cnt_y_p./ni) .* ni;
% prob_n = (cnt_y_n./ni) .* ni;


likehood_p = (cnt_y_p./ni) ; % P(positive)
likehood_n = (cnt_y_n./ni) ; % P(negative)
if isFast == 0
    for j=1:length(Xi)
        likehood_p = likehood_p .* (c_p(Xi(j), j) ./ (cnt_y_p+1) );

        likehood_n = likehood_n .* (c_n(Xi(j), j) ./ (cnt_y_n+1) );
    end
elseif isFast ==1
    % number of total elements of the counting table.
    total_ele = numel(c_p);
    % number of features
    n_fe = length(Xi);
    n_rows = size(c_p, 1);
    tmp = 0:n_rows:(total_ele-1);    
    % idx = (Xi + 0:5:numel(c_n)),
    idx = (Xi + tmp )';
    c_p_1d = c_p(:);
    c_n_1d = c_n(:);
        
    Xi_p_cnt = c_p_1d(idx);
    Xi_n_cnt = c_n_1d(idx);
    
    likehood_p = likehood_p .* prod(Xi_p_cnt) / (cnt_y_p + 1)^n_fe;
    likehood_n = likehood_n .* prod(Xi_n_cnt) / (cnt_y_n + 1)^n_fe;
    
end


% prob_p = prob_p .* ni;
% prob_n = prob_n .* ni;
prob_p = likehood_p ./ (likehood_p + likehood_n);
prob_n = likehood_n ./ (likehood_p + likehood_n);


end

