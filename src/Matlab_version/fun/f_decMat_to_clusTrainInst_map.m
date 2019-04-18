function clusMap = f_decMat_to_clusTrainInst_map(dec_mat, adv_whole, adv_self)
% input the decmat, output the mapping from cluster to the proper training
%  cluster.

if nargin < 2
% adv_whole = 0.15;
% adv_self = 0.08;
adv_whole = 0.2;
adv_self = 0.15;
end

% deal with NaN
[dec_mat] = f_proc_dec_mat(dec_mat);

n_clus = size(dec_mat, 2);
clusMap = (1:n_clus)';
for i=1:n_clus
    dec_col = dec_mat(:, i);
    
    % give some davantage to whole and self.
    dec_col(end) = dec_col(end) + adv_whole;
    dec_col(i) = dec_col(i) + adv_self;
    
    [~, ix] = sort(dec_col, 'descend');
    clus_id_to_use = ix(1);
    clusMap(i, 2) = clus_id_to_use;
end

% convert the (n_clus+1) in the 2nd col to 0. (Use 0 as the code for whole
%  data)
a = clusMap(:, 2);
a(a == n_clus+1) = 0;
clusMap(:, 2) = a;


end

