function [ y_preds ] = f_NB_leaveOneOut(X, y, c_p, c_n, cnt_y_p, cnt_y_n)
% Predict class labels using NB, using leave one out.

y_preds = zeros(size(y));
for i = 1:length(y)
    Xi = X(i, :);
    yi = y(i);
    y_pred = f_NB_pred(c_p, c_n, cnt_y_p, cnt_y_n, Xi, yi);
    y_preds(i) = y_pred;
end



end

