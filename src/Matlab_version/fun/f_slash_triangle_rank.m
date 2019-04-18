function [ a ] = f_slash_triangle_rank(a)
% 

% more neighbors priority or top neighbors' neighborhood first. 
neighbor_1st = 1;

if neighbor_1st == 0
    a = a';
end

a = repmat(1:size(a, 2), [size(a,1), 1]) + ...
    repmat((1:size(a, 1))', [1, size(a,2)]) ;
%%
[~, ix] = sort(a(:));
a(ix) = 1:numel(a);

if neighbor_1st == 0
    a = a';
end


% i=1; j=1; cur=1;
% [a, i, j, cur] = f_fill_a_slash(a, i, j, cur),
% while true
%     if j+1 <= 
% end
% 
% end

end


% function [a, i, j, cur] = f_fill_a_slash(a, i, j, cur)
% 
% a(i, j) = cur;
% cur = cur + 1;
% 
% while i+1 <= size(a, 1) && j-1 >= 1
%     i = i+1;
%     j = j-1;
%     a(i, j) = cur;
%     cur = cur + 1;
% end
% 
% 
% end


% exmaple:

% input 
%      0     0     0
%      0     0     0
%      0     0     0
%   
% if neighbor_1st == 0
% output
%      1     2     4
%      3     5     7
%      6     8     9

% if neighbor_1st == 1
% output
%      1     3     6
%      2     5     8
%      4     7     9









