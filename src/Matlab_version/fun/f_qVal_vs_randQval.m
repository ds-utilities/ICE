function [ qval, qval_rands, q_diff ] = ...
    f_qVal_vs_randQval(X, k, rankType, n_times, simType)
% X is the data
% k is the kNN (aRank) number.

% typ: type of the original kNN graph. Could be 'arank' (best neighbors of 
%   each node), 'mrank' (top mutual connection), 'com' (combined mRank and 
%   1-NN aRank)

% n_times is the number of times of randomization.
%  randomize the data for 20 times and compute the control q_value

qval = f_data_2_graph(X, k, rankType, 0, simType);

qval_rands = zeros(1, n_times);
for i=1:n_times
    qval_rands(i) = f_data_2_graph(X, k, rankType, 1, simType);
end

q_diff = qval - mean(qval_rands);

end

