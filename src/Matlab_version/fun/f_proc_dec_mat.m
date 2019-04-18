function [dec_mat] = f_proc_dec_mat(dec_mat)
% deal with NaN in dec_mat

for i= 1: size(dec_mat, 2)
    dec_col = dec_mat(:, i);
    
    % if the col is all nan, then choose whole
    if sum(isnan(dec_col) == length(dec_col) )
        dec_col = zeros(legnth(dec_col), 1);
        dec_col(end) = 1;
    end
    
    % if whole is NaN, then choose whole
    if isnan( dec_col(end) )
       dec_col(end) = 1;
    end
    
    % if other has NaN, then don't choose it.
    dec_col( isnan(dec_col) ) = -1;
    
    dec_mat(:, i) = dec_col;
    
end



end

