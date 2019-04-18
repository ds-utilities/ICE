function [tfs] = f_clus_to_tfs(clus, n_inst)
% convert the cluster information from cell array to mat. But for each
%  instance, the rank of clusters information will be lost - you won't know
%  what is the top 1/2/3 cluster it belongs to. 
% 
% clus e.g:
% 1x5 cell
% 1x195 double  1x193 double  1x169 double  1x161 double 1x62 double
%
% tfs e.g:
% 295x5 double
%      1     0     0     0     0
%      1     1     1     1     0
%      1     1     1     0     0
%      1     1     0     0     0
%      1     1     1     1     0
% ...
%      1     1     1     1     1
%      1     0     0     0     0
%      1     1     1     0     0

tfs = false(n_inst, length(clus));
for i=1:length(clus)
    tfs(clus{i}, i) = true;
end




end

