function [clus_new ] = f_get_reverse_clus(clus, n_inst)
% 
n_clus = length(clus);
clus_new = clus;

for i=1:n_clus
    ids = clus{i};
    tf = false(n_inst, 1);
    tf(ids) = true;
    ids_reverse = find( ~tf );
    clus_new{n_clus+i} = ids_reverse;
end



end

