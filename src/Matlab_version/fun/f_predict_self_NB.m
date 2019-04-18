function [prob_p, prob_n]=f_predict_self_NB(c_p, c_n, cnt_y_p, cnt_y_n,...
    X_test)
% 
n_insts = size(X_test, 1);
prob_p = zeros(n_insts, 1);
prob_n = zeros(n_insts, 1);

for i=1:n_insts
    Xi = X_test(i, :);
    [pp, pn] = f_NB_pred_2(c_p, c_n, cnt_y_p, cnt_y_n,  Xi);
    % c_p, c_n, cnt_y_p, cnt_y_n,
    
    prob_p(i) = pp;
    prob_n(i) = pn;
end


end

