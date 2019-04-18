function szs = f_gen_neighbor_count_cutoff_2(n_nodes)
% 

if n_nodes < 90
    szs = f_gen_neighbor_count_cutoff(n_nodes);
else % >= 90
    point_1 = 50;
    n_rest = n_nodes - point_1;
    szs = f_gen_neighbor_count_cutoff(n_rest, 5, 2);
    szs = [point_1, szs + point_1];
    
end



end

