function [ res, q_diffs, q_diff_max, k_best ] = ...
    f_qVal_vs_randQval_ks(X, ks, rankType, n_times, simType)
% X is the data
% ks is the kNN (aRank) numbers.
% n_times is the number of times of randomization.

% typ: type of the original kNN graph. Could be 'arank' (best neighbors of 
%   each node), 'mrank' (top mutual connection), 'com' (combined mRank and 
%   1-NN aRank)

% n_times is the number of times of randomization.
%  randomize the data for 20 times and compute the control q_value

q_diffs = zeros(length(ks), 1);

res = {};
for i=1:length(ks)
    [ qval, qval_rands, q_diff ] = ...
        f_qVal_vs_randQval(X, ks(i), rankType, n_times, simType);
    res{i, 1} = ks(i);
    res{i, 2} = qval;
    res{i, 3} = qval_rands;
    res{i, 4} = q_diff;
    
    q_diffs(i) = q_diff;
end

q_diff_max = max(q_diffs);
best_ixs = find(q_diffs == q_diff_max);
k_best = ks(best_ixs(1));

end

