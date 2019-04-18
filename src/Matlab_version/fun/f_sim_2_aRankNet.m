function aRankNet = f_sim_2_aRankNet(sim, k)
% Convert the similarity matrix to a network graph where each node 
%  has k edges to other nodes.
% A rank KNN graph
if nargin < 2
    k=3;
end

% delete the diagnal values.
sim = sim-diag(diag(sim) );

[~, I] = sort(sim-diag(diag(sim) ) );
% c is the index of column, i is the index for row
% For each column, the I_ic means sim_(I_ic) is at rank i.

[~, I2] = sort(I);
% I2_ic means sim_ic is at rank I2_ic.

% zhen: for every column, just keep the top k edges.
aRankNet = (I2 >length(sim)-k);

% make it a diagonal matrix
aRankNet = max(aRankNet, aRankNet');
% remove the diagonal 1s.
aRankNet = aRankNet-diag(diag(aRankNet) );

end

% revised from function f_gen_NN1_net(sim) at 
%  /Users/zhengao/Dropbox/mylab/MATLAB_lab/rwr_ensemble/

%
% sim =
% 
%    [1.0000    0.5566    0.6448    0.3289
%     0.5566    1.0000   -0.0842   -0.0170
%     0.6448   -0.0842    1.0000    0.8405
%     0.3289   -0.0170    0.8405    1.0000]
%     
% sim-diag(diag(sim)) = 
% 
%          0    0.5566    0.6448    0.3289
%     0.5566         0   -0.0842   -0.0170
%     0.6448   -0.0842         0    0.8405
%     0.3289   -0.0170    0.8405         0
% 
% 
% I =
% 
%      1     3     2     2
%      4     4     3     4
%      2     2     1     1
%      3     1     4     3
% 
% 
% I2 =
% 
%      1     4     3     3
%      3     3     1     1
%      4     1     2     4
%      2     2     4     2
% 
% 
% aRankNet =
% 
%      0     1     1     0
%      1     0     0     0
%      1     0     0     1
%      0     0     1     0 (when k = 1)
%
% aRankNet =
% 
%      0     1     1     0
%      1     0     0     0
%      1     0     0     1
%      0     0     1     0 (when k = 2)
%
% aRankNet =
%      0     1     1     1
%      1     0     0     1
%      1     0     0     1
%      1     1     1     0 (when k = 3)


