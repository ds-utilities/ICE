function [q_score, train, test] = f_arffPreProc_for_weka(pa, name, k_fold,...
    neighb_ver, low_per, high_per,  rankType, simType)
% ARFF data pre-processign for weka machine learning.
% 
% Read arff file to MATLAB
% zscore transformation
% Generate 20 arff files for 10-fold cross-validation. (train and test)
% Generate random-walk networks for PC_classifiers.
%
% low_per:  neighborhood percentage at least to the whole instances.
% high_per: neighborhood percentage at  most to the whole instances.
%
% rankType: 'aRank', 'mRank', 'mRankAuto' etc. Currently 'mRank' means auto
%  mRank, aRank is 3NN aRank.
%
% simType: 'euclidean', 'cosine', 'IDFsim' etc.

if nargin < 7
    rankType = 'mRank'; % 'mRank' means auto mRank
end
if nargin < 8
    simType = 'euclidean'; % 
end

if nargin < 4
    % neighb_ver = 3;
    neighb_ver = 2;
    low_per = 0;
    high_per = 0;
    % low_per = 2/5;
    % high_per = 4/5;
end
% ------------------------------------------------------------------------
%% Read arff file to MATLAB
% fn_whole = [pa, name, '.arff'];
% [X, y] = f_read_arff_w_labels(fn_whole);

% zscore transformation
%X = zscore(X);
% since all data in that paper has already been normalized, we do not need
% to do zscore() here. 

% save([pa, name, '.mat'], 'X', 'y');
%% convert the mat file to arff file
% update in 10-12 2017
% the original data set folders from the ref paper usually has the arff
%  file, but sometimes don't, and the arff files are the data done with
%  problematic data pre-processing, such as nominal data treatment. Here I
%  have already correct the data in the mat file, I need to correct the
%  data in the arff file now. note that the arff files have been backuped
%  already in the same folder.

fn_whole = [pa, name, '.arff'];
fn_whole_bk = [pa, name, '_bk.arff'];
system(['mv ', fn_whole, ' ', fn_whole_bk]);

fn_mat = [pa, name, '.mat'];
load(fn_mat);

[head, ~] = f_read_arff_2_strs(fn_whole_bk);
f_gen_arff_forClass_2(fn_whole, head, X, y);

%% Generate arff files for k-fold cross-validation.
% Since the 10-fold arff files are very stable, not like the rw net files, 
%  if it has already been generated, we can skip it. 
% if exist([pa, 'arff_kf/', 'fo_10_te.arff'], 'file') ~= 2
    [head, ~] = f_read_arff_2_strs(fn_whole);
    [train, test] = f_X_2_kFold_arffs(X, y, pa, head, k_fold);
% end

%% Generate RW random walk net files.
% use 3NN aRank
% q_score = f_X_2_rwNetFiles(X, pa, k_fold, neighb_ver, low_per, high_per);
% use auto kNN mRank with 1NN aRank combined original graph.
% if exist([pa, 'rwNets/', 'net_fo_10_id_1.mat'], 'file') ~= 2
  q_score = f_X_2_rwNetFiles_2(X, pa, k_fold, neighb_ver,low_per,high_per,...
      rankType, simType, train, test);
% end


end





