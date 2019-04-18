function newC = f_shrink_clusters(c, k)
% Combine the clusters with less than k elements into one cluster
% Sometimes such as when I need to use each cluster as a group of
%  attributes to train a regression model, if there is only 1 attribute,
%  the result won't be good. In this case, the potentially good attribute
%  is wasted. So I just want to combine such attributes into one cluster. 
% If there is still some clusters with less than k elements, I will jsut
%  give up, since there is just around 1 cluster with less than k elements.

if nargin <2
    k=5;
end

newC = {};

% b: a big bag of the small clusters
b=[];
j=1;
for i=1:length(c)
    tmp = c{i};
    tmp = tmp(:);
    if length(tmp) <= k
        b = [b; tmp];
    else
        newC{j} = tmp;
        j=j+1;
    end
end
newC{j} = b; 

end

