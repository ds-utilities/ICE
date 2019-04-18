function [cnts_p, cnts_n] = f_cnt_freq_NB_col(col, y, uniqueVs)
    %uniqueV = unique(col);
    ixs_1 = y==1;
    ixs_0 = ~ixs_1;
    
    col_p = col(ixs_1);
    col_n = col(ixs_0);
    
    cnts_p = zeros(length(uniqueVs), 1);
    cnts_n = zeros(length(uniqueVs), 1);
    for i=1:length(uniqueVs)
        cnts_p(i) = sum(col_p == uniqueVs(i));
        cnts_n(i) = sum(col_n == uniqueVs(i));
    end
    
    
end