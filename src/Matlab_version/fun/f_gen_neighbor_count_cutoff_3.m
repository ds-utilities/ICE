function szs = f_gen_neighbor_count_cutoff_3(n_nodes, low_per, high_per)
% this version will assign more neighbors. The neighborhood size growth is
% linear.

% low_per:  neighborhood percentage at least to the whole instances.
% high_per: neighborhood percentage at  most to the whole instances.

% if nargin < 2
%     low_per = 2/5;
%     high_per = 4/5;
% end

szs1 = floor(n_nodes * low_per);
szsend = floor(n_nodes * high_per);

interval = floor( (szsend - szs1)/4 );
szs = szs1:interval:szsend;

end