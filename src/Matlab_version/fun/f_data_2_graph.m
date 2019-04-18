function [ q, graph ] = f_data_2_graph(X, n_NN, rankType, rand_data, simType)
% Convert the original data to a inst.-inst. similarity matrix using
%  IDFsim(), then use aRank to convert the sim matrix to a graph. 

% X: data
% n_NN: number of nearest neighbors when convert sim to graph.
% typ: type of the original kNN graph. Could be 'arank' (best neighbors of 
%   each node), 'mrank' (top mutual connection), 'com' (combined mRank and 
%   1-NN aRank)

% rand_data: 1 - randomize the data; 0 - don't randomize the data.


% if nargin < 4
%     rand_data = 0;
% end

%% randomdize the data
if rand_data
    X = f_randPerm_data(X);
end

%% distance / similarity computing
if strcmp(simType, 'IDFsim')
    sim = IDFsim(X);
elseif strcmp(simType, 'cosine')
    sim = squareform(pdist(X, 'cosine')); sim = -sim;
elseif strcmp(simType, 'euclidean')
    sim = squareform(pdist(X)); sim = -sim;
elseif strcmp(simType, 'corr')
    sim = squareform(pdist(X, 'correlation')); sim = -sim;
end
% 

%% rank type
if strcmp(rankType, 'aRank')
    graph = f_sim_2_aRankNet(sim, n_NN);
elseif strcmp(rankType, 'mRank')
    graph = f_sim_2_mRankNet(sim, n_NN);
    graph = graph + f_gen_NN1_net(sim);
% elseif strcmp(rankType, 'com')
%     graph = f_sim_2_mRankNet(sim, n_NN);
%     graph = graph + f_gen_NN1_net(sim);
end

%% Qcut cuts a graph to several clusters.
[~, q] = Qcut(graph);


end


% related function:
% getAutoGeometricCombinedNetFromSim()



