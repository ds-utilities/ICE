function [each_clus_szs] = f_clus_size(n)
% given the number of instances, n
% get the clusters size from small to big
% s1 = 16; % the smallest cluster size is 16 
s1 = 32; % the smallest cluster size is 16 
% incr = 2; % the cluster size incremental factor. 
incr = 2; % the cluster size incremental factor. 
% perc = 3/4; % the largest cluster proportion
perc = 4/5; % the largest cluster proportion

while s1(end) < floor(n/2)
    s1 = [s1, s1(end) * incr ];
    
    if s1(end) > ceil( n * perc )
        break;
    end
    if s1(end) < ceil( n * perc ) && s1(end) > ceil(n/2)
        s1 = [s1, ceil( n * perc )];
        break;
    end
end

each_clus_szs = s1;


end

