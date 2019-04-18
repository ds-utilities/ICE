function [Y_predicteds] = f_predict_randForest_MATLAB(X, m)
% prediction using the MATLAB version of random forest.

Y_predicteds = predict(m, X);
[tf , ~] = ismember(Y_predicteds, '1');

Y_predicteds = tf;

end